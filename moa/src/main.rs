//! `moa` command-line interface.

use moa_core::backend::{Backend, Engine};
use moa_core::minhash::{GraphParams, SketchParams};
use moa_core::{art, cluster_labeled_graph, cluster_sequences, MclParams};
use std::collections::BTreeSet;
use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{}", art::banner("Run `moa help` for usage."));
        exit(2);
    }
    let cmd = args[0].as_str();
    let rest = &args[1..];
    let code = match cmd {
        "mcl" => cmd_mcl(rest),
        "cluster" => cmd_cluster(rest),
        "doctor" => cmd_doctor(rest),
        "info" => cmd_info(),
        "version" | "--version" | "-V" => {
            println!("moa v{}  {}", art::VERSION, art::MOA_MINI);
            0
        }
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("moa: unknown command '{other}'. Try `moa help`.");
            2
        }
    };
    exit(code);
}

/// Shared options parsed from the tail of a command line.
struct Opts {
    positional: Vec<String>,
    inflation: f64,
    engine: Engine,
    output: Option<String>,
    prune: usize,
    select: usize,
    recover: usize,
    max_iter: usize,
    // sketch / graph (cluster only)
    k: usize,
    s: usize,
    min_id: f64,
    len_cov: f64,
    quiet: bool,
}

impl Default for Opts {
    fn default() -> Self {
        let d = MclParams::default();
        Opts {
            positional: Vec::new(),
            inflation: d.inflation,
            engine: Engine::Rayon,
            output: None,
            prune: d.prune,
            select: d.select,
            recover: d.recover,
            max_iter: d.max_iter,
            k: 7,
            s: 256,
            min_id: 0.95,
            len_cov: 0.70,
            quiet: false,
        }
    }
}

impl Opts {
    fn mcl_params(&self) -> MclParams {
        MclParams {
            inflation: self.inflation,
            prune: self.prune,
            select: self.select,
            recover: self.recover,
            max_iter: self.max_iter,
            ..MclParams::default()
        }
    }
    fn graph_params(&self) -> GraphParams {
        GraphParams {
            sketch: SketchParams {
                k: self.k,
                s: self.s,
            },
            min_id: self.min_id,
            len_cov: self.len_cov,
            ..GraphParams::default()
        }
    }
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts::default();
    let mut i = 0;
    let need = |i: &usize, args: &[String], flag: &str| -> Result<String, String> {
        args.get(i + 1)
            .cloned()
            .ok_or_else(|| format!("option {flag} needs a value"))
    };
    let pf = |s: &str, flag: &str| -> Result<f64, String> {
        s.parse::<f64>().map_err(|_| format!("{flag}: '{s}' is not a number"))
    };
    let pu = |s: &str, flag: &str| -> Result<usize, String> {
        s.parse::<usize>().map_err(|_| format!("{flag}: '{s}' is not an integer"))
    };
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-I" | "--inflation" => {
                o.inflation = pf(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "--engine" => {
                let v = need(&i, args, a)?;
                o.engine = Engine::parse(&v)
                    .ok_or_else(|| format!("--engine: unknown engine '{v}' (serial|rayon|hydra|gpu|auto)"))?;
                i += 1;
            }
            "-o" | "--output" => {
                o.output = Some(need(&i, args, a)?);
                i += 1;
            }
            "-P" | "--prune" => {
                o.prune = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "-S" | "--select" => {
                o.select = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "-R" | "--recover" => {
                o.recover = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "--max-iter" => {
                o.max_iter = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "-k" => {
                o.k = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "-s" => {
                o.s = pu(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "--min-id" => {
                o.min_id = pf(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "--len-cov" => {
                o.len_cov = pf(&need(&i, args, a)?, a)?;
                i += 1;
            }
            "-q" | "--quiet" => o.quiet = true,
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option '{s}'"));
            }
            _ => o.positional.push(a.to_string()),
        }
        i += 1;
    }
    Ok(o)
}

/// Order clusters largest-first (ties by first member) and render one per line,
/// members tab-separated — the same shape `mcl -o` emits.
fn format_clusters(clusters: &[Vec<String>]) -> String {
    let mut idx: Vec<usize> = (0..clusters.len()).collect();
    idx.sort_by(|&a, &b| {
        clusters[b]
            .len()
            .cmp(&clusters[a].len())
            .then_with(|| clusters[a].first().cmp(&clusters[b].first()))
    });
    let mut out = String::new();
    for &c in &idx {
        out.push_str(&clusters[c].join("\t"));
        out.push('\n');
    }
    out
}

fn write_out(output: &Option<String>, text: &str) -> Result<(), String> {
    match output {
        None => {
            print!("{text}");
            Ok(())
        }
        Some(path) => std::fs::write(path, text).map_err(|e| format!("writing {path}: {e}")),
    }
}

fn start_backend(engine: Engine) -> Backend {
    match Backend::start(engine) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("moa: cannot start '{}' engine: {e}", engine.as_str());
            exit(1);
        }
    }
}

fn cmd_mcl(args: &[String]) -> i32 {
    let o = match parse_opts(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("moa mcl: {e}");
            return 2;
        }
    };
    if o.positional.is_empty() {
        eprintln!("moa mcl: need a graph file (label pairs / .abc). See `moa help`.");
        return 2;
    }
    let path = &o.positional[0];
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("moa mcl: reading {path}: {e}");
            return 1;
        }
    };
    let backend = start_backend(o.engine);
    let t0 = Instant::now();
    let (clusters, stats) = cluster_labeled_graph(&text, &o.mcl_params(), &backend);
    let dt = t0.elapsed();
    if !o.quiet {
        eprintln!(
            "moa mcl [{}]: {} nodes, {} edges ({} nnz) -> {} clusters in {} iters (chaos {:.2e}, {:.3} ms)",
            backend.describe(),
            stats.n_nodes,
            stats.n_edges,
            stats.nnz,
            clusters.len(),
            stats.iterations,
            stats.final_chaos,
            dt.as_secs_f64() * 1e3,
        );
    }
    if let Err(e) = write_out(&o.output, &format_clusters(&clusters)) {
        eprintln!("moa mcl: {e}");
        return 1;
    }
    0
}

fn cmd_cluster(args: &[String]) -> i32 {
    let o = match parse_opts(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("moa cluster: {e}");
            return 2;
        }
    };
    if o.positional.is_empty() {
        eprintln!("moa cluster: need a FASTA file. See `moa help`.");
        return 2;
    }
    let path = &o.positional[0];
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("moa cluster: reading {path}: {e}");
            return 1;
        }
    };
    let backend = start_backend(o.engine);
    let t0 = Instant::now();
    let r = cluster_sequences(&text, &o.graph_params(), &o.mcl_params(), &backend);
    let dt = t0.elapsed();
    if !o.quiet {
        eprintln!(
            "moa cluster [{}]: {} sequences, {} edges (k={}, s={}, min-id={:.2}) -> {} clusters in {} iters (chaos {:.2e}, {:.3} ms)",
            backend.describe(),
            r.n_seqs,
            r.n_edges,
            o.k,
            o.s,
            o.min_id,
            r.clusters.len(),
            r.iterations,
            r.final_chaos,
            dt.as_secs_f64() * 1e3,
        );
    }
    if let Err(e) = write_out(&o.output, &format_clusters(&r.clusters)) {
        eprintln!("moa cluster: {e}");
        return 1;
    }
    0
}

/// Partition the friends graph as a canonical set-of-sets for comparison.
fn friends_partition(engine: Engine) -> Result<BTreeSet<BTreeSet<String>>, String> {
    let backend = Backend::start(engine)?;
    let (clusters, _) =
        cluster_labeled_graph(moa_core::FRIENDS_ABC, &MclParams::default(), &backend);
    Ok(clusters
        .into_iter()
        .map(|c| c.into_iter().collect::<BTreeSet<_>>())
        .collect())
}

fn cmd_doctor(_args: &[String]) -> i32 {
    print!("{}", art::banner("self-test: every engine vs. serial on the friends graph"));

    // Reference partition from the serial engine.
    let reference = match friends_partition(Engine::Serial) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("doctor: serial reference failed: {e}");
            return 1;
        }
    };
    let expected_3 = reference.len() == 3;
    println!("{}", art::rule("expansion engines"));
    println!(
        "  reference (serial): {} clusters at -I 2.0  [{}]",
        reference.len(),
        if expected_3 { "expected 3" } else { "UNEXPECTED" }
    );

    let engines = [
        ("serial", Engine::Serial),
        ("rayon", Engine::Rayon),
        ("hydra", Engine::Hydra),
        ("gpu", Engine::Gpu),
    ];
    let mut all_ok = expected_3;
    for (name, eng) in engines {
        let t0 = Instant::now();
        match Backend::start(eng) {
            Err(e) => {
                println!("  {name:<7} UNAVAILABLE  ({e})");
                if eng == Engine::Gpu {
                    // A missing GPU is not a failure of moa itself.
                } else {
                    all_ok = false;
                }
            }
            Ok(backend) => {
                let desc = backend.describe();
                drop(backend);
                match friends_partition(eng) {
                    Ok(part) => {
                        let dt = t0.elapsed().as_secs_f64() * 1e3;
                        let same = part == reference;
                        // GPU is f32: accept a matching partition even if not bit-identical.
                        let ok = same;
                        println!(
                            "  {name:<7} {:<4} {} clusters, {:.3} ms   [{}]",
                            if ok { "OK" } else { "FAIL" },
                            part.len(),
                            dt,
                            desc
                        );
                        if !ok {
                            all_ok = false;
                        }
                    }
                    Err(e) => {
                        println!("  {name:<7} ERROR  ({e})");
                        all_ok = false;
                    }
                }
            }
        }
    }

    println!("\n{}", art::rule("granularity check"));
    // -I 1.4 must coarsen the friends graph to two groups.
    let backend = Backend::start(Engine::Serial).unwrap();
    let coarse = MclParams {
        inflation: 1.4,
        ..MclParams::default()
    };
    let (c14, s14) = cluster_labeled_graph(moa_core::FRIENDS_ABC, &coarse, &backend);
    let ok14 = c14.len() == 2;
    println!(
        "  -I 1.4 -> {} clusters (chaos {:.2e})  [{}]",
        c14.len(),
        s14.final_chaos,
        if ok14 { "expected 2" } else { "UNEXPECTED" }
    );
    if !ok14 {
        all_ok = false;
    }

    println!(
        "\n  {}  {}",
        if all_ok { "ALL CHECKS PASSED" } else { "SOME CHECKS FAILED" },
        art::MOA_MINI
    );
    if all_ok {
        0
    } else {
        1
    }
}

fn cmd_info() -> i32 {
    print!("{}", art::banner("build & environment"));
    println!("{}", art::rule("build"));
    println!("  version      moa v{}", art::VERSION);
    println!("  language     100% Rust (edition 2021, MSRV 1.75)");
    println!(
        "  gpu feature  {}",
        if cfg!(feature = "gpu") { "enabled (wgpu)" } else { "disabled" }
    );
    println!("  license      CC-BY-NC-4.0");

    println!("{}", art::rule("expansion engines"));
    println!("  serial   always available (correctness reference)");
    println!(
        "  rayon    available ({} threads)",
        rayon::current_num_threads()
    );
    match Backend::start(Engine::Hydra) {
        Ok(b) => {
            println!("  hydra    available  [{}]", b.describe());
        }
        Err(e) => println!("  hydra    unavailable  ({e})"),
    }
    match Backend::start(Engine::Gpu) {
        Ok(b) => println!("  gpu      available  [{}]", b.describe()),
        Err(e) => println!("  gpu      unavailable  ({e})"),
    }
    0
}

fn print_help() {
    print!("{}", art::banner("Markov clustering, faithfully; and MinHash graphs, quickly."));
    println!(
        r#"USAGE
  moa <command> [options]

COMMANDS
  mcl <graph>       Cluster a weighted graph given as labelled edges
                    ("labelA labelB [weight]" per line; .abc / .mci-abc style).
  cluster <fasta>   Build a MinHash similarity graph over the sequences in a
                    FASTA file, then cluster it with MCL. Alignment-free.
  doctor            Self-test: run the friends example on every expansion
                    engine and check they agree with the serial reference.
  info              Show build, feature, and backend/environment details.
  version           Print the version.
  help              Print this help.

COMMON OPTIONS
  -I, --inflation <f>   Inflation / granularity (default 2.0; higher = finer).
      --engine <e>      Expansion engine: serial | rayon | hydra | gpu | auto
                        (default rayon). All engines produce the same clusters.
  -o, --output <file>   Write clusters to a file (default: stdout).
  -P, --prune <n>       Prune number (default 10000; cut = 0.99999/n).
  -S, --select <n>      Select number: max entries kept per column (default 1100).
  -R, --recover <n>     Recover number (default 1400).
      --max-iter <n>    Iteration cap (default 10000; convergence is far sooner).
  -q, --quiet           Suppress the per-run summary on stderr.

CLUSTER-ONLY OPTIONS
  -k <n>                k-mer size for sketching (default 7).
  -s <n>                Bottom-s sketch size (default 256; larger = less noise).
      --min-id <f>      Minimum identity for an edge (default 0.95).
      --len-cov <f>     Minimum length ratio, shorter/longer (default 0.70).

OUTPUT
  One cluster per line, members tab-separated, largest cluster first.

EXAMPLES
  moa mcl friends.abc -I 2.0
  moa mcl graph.abc --engine gpu -o clusters.tsv
  moa cluster proteins.faa -I 1.8 --min-id 0.90
  moa doctor

  "{cat}"
"#,
        cat = art::CATCHPHRASE
    );
}
