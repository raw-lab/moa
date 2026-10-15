//! GPU expansion backend (wgpu compute).
//!
//! Expansion is a matrix square. On the GPU the cleanest formulation is a plain
//! dense product `C = A*A`: densify the (usually sparse) operand into an `n x n`
//! `f32` buffer, run a tiled GEMM compute shader, read the result back, and
//! re-sparsify by dropping the zeros. The per-column prune / select / recover
//! and the chaos measure then run on the CPU exactly as for every other engine,
//! so the *only* difference a GPU run can make is `f32` rounding — which never
//! moves the friends partition, and in practice never changes a clustering.
//!
//! wgpu speaks Vulkan, Metal, DX12 and GL, and will happily bind a software
//! adapter (Mesa's `lavapipe`/`llvmpipe`) when no discrete GPU exists, so this
//! path is exercisable on any machine. WGSL has no `f64`, hence the single
//! precision; the CPU engines remain the `f64` reference.
//!
//! Dense storage is `O(n^2)`, so above [`GPU_MAX_N`] this backend transparently
//! falls back to the rayon engine for that expansion (still correct, just not on
//! the GPU) and says so once.

use crate::matrix::SparseMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest `n` for which we build the dense `n x n` `f32` operand on-device.
/// 2048 keeps each buffer at ~16 MiB, well under wgpu's 128 MiB default storage
/// binding limit, with headroom for the two operands.
pub const GPU_MAX_N: usize = 2048;

const SHADER: &str = r#"
struct Dims { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dims.n;
    let i = gid.x;
    let j = gid.y;
    if (i >= n || j >= n) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        acc = acc + a[i * n + k] * a[k * n + j];
    }
    c[i * n + j] = acc;
}
"#;

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    name: String,
    backend: wgpu::Backend,
    warned_fallback: AtomicBool,
}

impl GpuBackend {
    pub fn start() -> Result<GpuBackend, String> {
        pollster::block_on(Self::start_async())
    }

    async fn start_async() -> Result<GpuBackend, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Prefer Vulkan (covers lavapipe); allow GL as a further fallback.
            backends: wgpu::Backends::VULKAN
                | wgpu::Backends::METAL
                | wgpu::Backends::DX12
                | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| {
                "no GPU adapter found (no Vulkan/Metal/DX12/GL device, and no software \
                 lavapipe). Install a Vulkan driver or use `--engine rayon`."
                    .to_string()
            })?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("moa.gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("GPU device request failed: {e}"))?;

        // Capture validation errors instead of letting wgpu's default handler
        // abort the process, so a hostile driver surfaces as a clean `Err`.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moa.expand.gemm"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moa.expand.pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });
        if let Some(err) = device.pop_error_scope().await {
            return Err(format!("GPU shader/pipeline validation failed: {err}"));
        }

        Ok(GpuBackend {
            device,
            queue,
            pipeline,
            name: info.name,
            backend: info.backend,
            warned_fallback: AtomicBool::new(false),
        })
    }

    pub fn describe(&self) -> String {
        format!("wgpu {:?} — {} (f32, dense GEMM, n<={})", self.backend, self.name, GPU_MAX_N)
    }

    pub fn expand(&self, m: &SparseMatrix) -> SparseMatrix {
        let n = m.n;
        if n == 0 {
            return SparseMatrix::new(0);
        }
        if n > GPU_MAX_N {
            if !self.warned_fallback.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "moa: graph has {n} nodes (> GPU dense cap {GPU_MAX_N}); \
                     expanding on CPU (rayon) instead. Use `--engine rayon` to silence this."
                );
            }
            return super::cpu::expand_rayon(m);
        }
        pollster::block_on(self.expand_async(m))
    }

    async fn expand_async(&self, m: &SparseMatrix) -> SparseMatrix {
        use wgpu::util::DeviceExt;
        let n = m.n;
        let nn = n * n;

        // Densify column-major matrix into a row-major f32 buffer: a[i*n + j].
        let mut a = vec![0.0f32; nn];
        for (j, col) in m.cols.iter().enumerate() {
            for &(i, v) in col.iter() {
                a[i as usize * n + j] = v as f32;
            }
        }

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("moa.a"),
                contents: bytemuck::cast_slice(&a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_size = (nn * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moa.c"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta = [n as u32, 0, 0, 0];
        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("moa.meta"),
                contents: bytemuck::cast_slice(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moa.staging"),
            size: c_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moa.bind"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("moa.enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("moa.expand.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = ((n as u32) + 7) / 8;
            pass.dispatch_workgroups(groups, groups, 1);
        }
        enc.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, c_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU map channel closed")
            .expect("GPU buffer map failed");

        let data = slice.get_mapped_range();
        let c: &[f32] = bytemuck::cast_slice(&data);

        // Re-sparsify: column j gathers rows i with a nonzero product, in
        // ascending row order (which is exactly how we scan i).
        let mut out = SparseMatrix::new(n);
        for j in 0..n {
            let mut col = Vec::new();
            for i in 0..n {
                let v = c[i * n + j];
                if v != 0.0 {
                    col.push((i as u32, v as f64));
                }
            }
            out.cols[j] = col;
        }
        drop(data);
        staging.unmap();
        out
    }
}
