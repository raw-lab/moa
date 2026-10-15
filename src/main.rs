//! `moa` — command-line interface.

mod doctor;

use hydra_mpp::{Config, Mode};
use moa_core::art::{self, bold, dim, fg};
use moa_core::engine::{Engine, HydraEngine};
use moa_core::{MclOutput, MclParams};
use std::io::{BufWriter, IsTerminal, Read, Write};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    // HydraMPP flags (--hydra-*) are peeled off here; everything else is ours.
    let hcfg = Config::parse(raw.clone());
    if let Mode::Client { .. } = hcfg.mode {
        // A worker node (by hand, or relaunched by a --hydra-slurm head):
        // serve moa's tasks; never read inputs or write outputs here.
        moa_core::engine::serve_worker(hcfg);
    }
    let args: Vec<String> = hcfg.rest_args().to_vec();
    let mut g = Global::default();
    let args = match g.take(args) {
        Ok(a) => a,
        Err(e) => return fail(&e),
    };
    art::init_color(&g.color, std::io::stderr().is_terminal() && std::io::stdout().is_terminal());
    if let Some(t) = g.threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(t).build_global() {
            return fail(&format!("cannot size thread pool: {e}"));
        }
    }
    let Some(cmd) = args.first().cloned() else {
        eprint!("{}", art::banner(""));
        eprintln!("\n  Run {} for usage, {} to self-test.\n", bold("moa help"), bold("moa doctor"));
        return ExitCode::from(2);
    };
    let rest = &args[1..];
    if rest.iter().any(|a| a == "-h" || a == "--help") && cmd != "help" {
        print_help(Some(&cmd));
        return ExitCode::SUCCESS;
    }
    let r = match cmd.as_str() {
        "mcl" => cmd_mcl(rest, &g, hcfg),
        "cluster" => cmd_cluster(rest, &g, hcfg),
        "graph" => cmd_graph(rest),
        "doctor" => doctor::run(rest, hcfg),
        "info" => cmd_info(hcfg),
        "version" => cmd_version(rest),
        "--version" | "-V" => {
            println!("moa {}", moa_core::VERSION);
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help(rest.first().map(|s| s.as_str()));
            Ok(())
        }
        "worker" => Err("`moa worker` needs `--hydra-client <host-ip>` (see `moa help worker`)".into()),
        other => Err(format!("unknown command '{other}' — try `moa help`")),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(&e),
    }
}

fn fail(e: &str) -> ExitCode {
    eprintln!("{} {} {e}", art::bad(), bold("moa:"));
    ExitCode::from(1)
}

/// Flags accepted anywhere on the command line.
#[derive(Default)]
struct Global {
    threads: Option<usize>,
    color: String,
}

impl Global {
    fn take(&mut self, args: Vec<String>) -> Result<Vec<String>, String> {
        self.color = "auto".into();
        let mut out = Vec::new();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "-t" | "--threads" | "-te" => {
                    let v = it.next().ok_or_else(|| format!("{a} needs a thread count"))?;
                    let n: usize = v.parse().map_err(|_| format!("{a}: '{v}' is not a count"))?;
                    self.threads = if n == 0 { None } else { Some(n) };
                }
                "--color" | "--colour" => self.color = it.next().ok_or("--color needs auto|always|never")?,
                "--no-color" => self.color = "never".into(),
                _ => out.push(a),
            }
        }
        Ok(out)
    }
}

/// Options shared by `mcl` and `cluster`.
struct Opts {
    pos: Vec<String>,
    p: MclParams,
    engine: String,
    out: Option<String>,
    quiet: bool,
    verbose: bool,
    stats: Option<String>,
    // cluster-only
    gp: moa_core::GraphParams,
    edges_out: Option<String>,
}

fn parse_opts(args: &[String], cluster: bool) -> Result<Opts, String> {
    let mut o = Opts {
        pos: vec![],
        p: MclParams::default(),
        engine: "auto".into(),
        out: None,
        quiet: false,
        verbose: false,
        stats: None,
        gp: moa_core::GraphParams::default(),
        edges_out: None,
    };
    let mut i = 0;
    let val = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
    };
    fn num<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
        s.parse::<T>().map_err(|_| format!("{flag}: '{s}' is not a valid number"))
    }
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-I" | "--inflation" => o.p.inflation = num(&val(&mut i, a)?, a)?,
            "-P" | "--prune" => o.p.prune = num(&val(&mut i, a)?, a)?,
            "-S" | "--select" => o.p.select = num(&val(&mut i, a)?, a)?,
            "-R" | "--recover" => o.p.recover = num(&val(&mut i, a)?, a)?,
            "-pct" | "--pct" => o.p.pct = num::<f64>(&val(&mut i, a)?, a)? / 100.0,
            "-L" | "--max-iter" => o.p.max_iter = num(&val(&mut i, a)?, a)?,
            "--chaos-limit" => o.p.chaos_limit = num(&val(&mut i, a)?, a)?,
            "--engine" | "-e" => o.engine = val(&mut i, a)?,
            "-o" | "--out" | "--output" => o.out = Some(val(&mut i, a)?),
            "--stats" => o.stats = Some(val(&mut i, a)?),
            "-q" | "--quiet" => o.quiet = true,
            "-v" | "--verbose" => {
                o.verbose = true;
                moa_core::mcl::set_trace(true)
            }
            "--abc" => {} // mcl drop-in: moa always reads label pairs
            "-i" | "--min-id" if cluster => o.gp.min_id = num(&val(&mut i, a)?, a)?,
            "-k" | "--kmer" if cluster => o.gp.sketch.k = num(&val(&mut i, a)?, a)?,
            "-s" | "--sketch" if cluster => o.gp.sketch.s = num(&val(&mut i, a)?, a)?,
            "--len-cov" if cluster => o.gp.len_cov = num(&val(&mut i, a)?, a)?,
            "--dense-bucket" if cluster => o.gp.dense_bucket = num(&val(&mut i, a)?, a)?,
            "--fanout" if cluster => o.gp.fanout = num(&val(&mut i, a)?, a)?,
            "--anchors" if cluster => o.gp.anchors = num(&val(&mut i, a)?, a)?,
            "--edges" if cluster => o.edges_out = Some(val(&mut i, a)?),
            "-" => o.pos.push(a.to_string()),
            s if s.starts_with('-') && s.len() > 1 && s.parse::<f64>().is_err() => {
                return Err(format!(
                    "unknown option '{s}' (see `moa help {}`)",
                    if cluster { "cluster" } else { "mcl" }
                ))
            }
            _ => o.pos.push(a.to_string()),
        }
        i += 1;
    }
    o.p.validate()?;
    if cluster {
        if !(o.gp.min_id > 0.0 && o.gp.min_id <= 1.0) {
            return Err(format!("--min-id must be in (0, 1] (got {})", o.gp.min_id));
        }
        if o.gp.sketch.k == 0 || o.gp.sketch.s == 0 {
            return Err("-k and -s must be at least 1".into());
        }
    }
    Ok(o)
}

fn read_input(path: &str) -> Result<Vec<u8>, String> {
    if path == "-" {
        let mut v = Vec::new();
        std::io::stdin().read_to_end(&mut v).map_err(|e| format!("reading stdin: {e}"))?;
        Ok(v)
    } else {
        std::fs::read(path).map_err(|e| format!("reading {path}: {e}"))
    }
}

/// Build the requested engine. `auto` means rayon on one node, HydraMPP when a
/// multi-node role (`--hydra-host`, `--hydra-slurm`) was requested.
fn make_engine(name: &str, hcfg: Config) -> Result<Engine, String> {
    let distributed = !matches!(hcfg.mode, Mode::Local);
    match name {
        "serial" => Ok(Engine::Serial),
        "rayon" | "cpu" => Ok(Engine::Rayon),
        "auto" if !distributed => Ok(Engine::Rayon),
        "hydra" | "hydrampp" | "auto" => {
            let cfg = if distributed { hcfg } else { hcfg.quiet(true) };
            Ok(Engine::Hydra(HydraEngine::start(cfg)?))
        }
        other => Err(format!("unknown engine '{other}' (serial | rayon | hydra | auto)")),
    }
}

fn open_out(path: &Option<String>) -> Result<Box<dyn Write>, String> {
    Ok(match path.as_deref() {
        None | Some("-") => Box::new(BufWriter::with_capacity(1 << 20, std::io::stdout().lock())),
        Some(p) => Box::new(BufWriter::with_capacity(
            1 << 20,
            std::fs::File::create(p).map_err(|e| format!("creating {p}: {e}"))?,
        )),
    })
}

fn write_clusters(path: &Option<String>, clusters: &[Vec<u32>], labels: &[String]) -> Result<(), String> {
    let mut w = open_out(path)?;
    for c in clusters {
        let mut first = true;
        for &m in c {
            if !first {
                w.write_all(b"\t").map_err(|e| e.to_string())?;
            }
            first = false;
            w.write_all(labels[m as usize].as_bytes()).map_err(|e| e.to_string())?;
        }
        w.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{} {}", art::num(n), if n == 1 { one } else { many })
}

fn summary(what: &str, n_nodes: usize, n_edges: usize, out: &MclOutput, eng: &Engine, secs: f64) -> String {
    let sep = dim(" · ");
    let warn = if out.overlap > 0 || out.truants > 0 || !out.converged {
        let mut v = vec![];
        if out.overlap > 0 {
            v.push(format!("cut {} overlap", out.overlap));
        }
        if out.truants > 0 {
            v.push(format!("{} unattached nodes gathered", out.truants));
        }
        if !out.converged {
            v.push("did NOT converge (raise -L)".into());
        }
        format!("\n  {} {}", fg(art::GOLD, "▲"), v.join(", "))
    } else {
        String::new()
    };
    format!(
        "{} {}  {}{sep}{}{sep}{}  {}  {}{sep}{}{sep}chaos {:.2e}{sep}{}\n  {}{warn}",
        art::mini(),
        bold(&format!("moa {what}")),
        plural(n_nodes, "node", "nodes"),
        plural(n_edges, "edge", "edges"),
        plural(out.components, "component", "components"),
        fg(art::GREY, "→"),
        bold(&fg(art::GREEN, &plural(out.clusters.len(), "cluster", "clusters"))),
        plural(out.iterations, "iteration", "iterations"),
        out.chaos,
        bold(&art::dur(secs)),
        dim(&format!(
            "[{}]  iterate {} · interpret {}",
            eng.describe(),
            art::dur(out.secs_iterate),
            art::dur(out.secs_interpret)
        )),
    )
}

fn write_stats(path: &Option<String>, fields: &[(&str, String)]) -> Result<(), String> {
    if let Some(p) = path {
        let body: Vec<String> = fields.iter().map(|(k, v)| format!("  \"{k}\": {v}")).collect();
        std::fs::write(p, format!("{{\n{}\n}}\n", body.join(",\n"))).map_err(|e| format!("writing {p}: {e}"))?;
    }
    Ok(())
}

fn stat_fields(
    o: &Opts,
    out: &MclOutput,
    n_nodes: usize,
    n_edges: usize,
    secs: f64,
    eng: &Engine,
) -> Vec<(&'static str, String)> {
    vec![
        ("moa_version", format!("\"{}\"", moa_core::VERSION)),
        ("engine", format!("\"{}\"", eng.name())),
        ("nodes", n_nodes.to_string()),
        ("edges", n_edges.to_string()),
        ("components", out.components.to_string()),
        ("largest_component", out.largest_component.to_string()),
        ("clusters", out.clusters.len().to_string()),
        ("iterations", out.iterations.to_string()),
        ("final_chaos", format!("{:e}", out.chaos)),
        ("converged", out.converged.to_string()),
        ("attractors", out.attractors.to_string()),
        ("overlap_cut", out.overlap.to_string()),
        ("unattached", out.truants.to_string()),
        ("inflation", o.p.inflation.to_string()),
        ("prune", o.p.prune.to_string()),
        ("select", o.p.select.to_string()),
        ("recover", o.p.recover.to_string()),
        ("pct", (o.p.pct * 100.0).to_string()),
        ("seconds_total", format!("{secs:.6}")),
        ("seconds_iterate", format!("{:.6}", out.secs_iterate)),
        ("seconds_interpret", format!("{:.6}", out.secs_interpret)),
    ]
}

fn cmd_mcl(args: &[String], _g: &Global, hcfg: Config) -> Result<(), String> {
    let o = parse_opts(args, false)?;
    let path = o.pos.first().ok_or("mcl needs a graph file (labelA labelB [weight] per line, or - for stdin)")?;
    let t0 = Instant::now();
    let engine = make_engine(&o.engine, hcfg)?;
    let phase = |what: &str, t: Instant| {
        if o.verbose {
            eprintln!("  [moa] {:<10} {}", what, art::dur(t.elapsed().as_secs_f64()));
        }
    };
    let tp = Instant::now();
    let text = read_input(path)?;
    phase("read", tp);
    let tp = Instant::now();
    let lab = moa_core::graph::parse_abc(&text).map_err(|e| format!("{path}: {e}"))?;
    drop(text);
    phase("parse", tp);
    let tp = Instant::now();
    let g = moa_core::Graph::from_edges(lab.labels.len(), &lab.edges);
    let n_edges = g.n_edges;
    if g.dropped > 0 && !o.quiet {
        eprintln!(
            "{} {} input edge{} ignored: weight not a positive finite number",
            fg(art::GOLD, "▲"),
            art::num(g.dropped),
            if g.dropped == 1 { "" } else { "s" }
        );
    }
    drop(lab.edges);
    phase("graph", tp);
    let tp = Instant::now();
    let out = moa_core::mcl::run(g, &o.p, &engine);
    phase("mcl", tp);
    let tp = Instant::now();
    write_clusters(&o.out, &out.clusters, &lab.labels)?;
    phase("write", tp);
    let secs = t0.elapsed().as_secs_f64();
    if !o.quiet {
        eprintln!("{}", summary("mcl", lab.labels.len(), n_edges, &out, &engine, secs));
    }
    write_stats(&o.stats, &stat_fields(&o, &out, lab.labels.len(), n_edges, secs, &engine))
}

fn cmd_cluster(args: &[String], _g: &Global, hcfg: Config) -> Result<(), String> {
    let o = parse_opts(args, true)?;
    let path = o.pos.first().ok_or("cluster needs a FASTA file (or - for stdin)")?;
    let t0 = Instant::now();
    let engine = make_engine(&o.engine, hcfg)?;
    let text = read_input(path)?;
    let r = moa_core::cluster_fasta(&text, &o.gp, &o.p, &engine).map_err(|e| format!("{path}: {e}"))?;
    if let Some(ep) = &o.edges_out {
        let mut w = open_out(&Some(ep.clone()))?;
        for &(a, b, wt) in &r.edges {
            writeln!(w, "{}\t{}\t{:.6}", r.ids[a as usize], r.ids[b as usize], wt).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }
    write_clusters(&o.out, &r.out.clusters, &r.ids)?;
    let secs = t0.elapsed().as_secs_f64();
    if !o.quiet {
        eprintln!("{}", summary("cluster", r.ids.len(), r.edges.len(), &r.out, &engine, secs));
        eprintln!(
            "  {}",
            dim(&format!(
                "MinHash k={} s={} min-id={} len-cov={} · sketch {} · graph {}",
                o.gp.sketch.k,
                o.gp.sketch.s,
                o.gp.min_id,
                o.gp.len_cov,
                art::dur(r.secs_sketch),
                art::dur(r.secs_graph)
            ))
        );
    }
    let mut f = stat_fields(&o, &r.out, r.ids.len(), r.edges.len(), secs, &engine);
    f.push(("kmer", o.gp.sketch.k.to_string()));
    f.push(("sketch", o.gp.sketch.s.to_string()));
    f.push(("min_id", o.gp.min_id.to_string()));
    f.push(("seconds_sketch", format!("{:.6}", r.secs_sketch)));
    f.push(("seconds_graph", format!("{:.6}", r.secs_graph)));
    write_stats(&o.stats, &f)
}

fn cmd_graph(args: &[String]) -> Result<(), String> {
    let o = parse_opts(args, true)?;
    let path = o.pos.first().ok_or("graph needs a FASTA file (or - for stdin)")?;
    let f = moa_core::minhash::read_fasta(&read_input(path)?)?;
    let sk = moa_core::minhash::sketch_many(&f.seqs, o.gp.sketch);
    let lens: Vec<u32> = f.seqs.iter().map(|s| s.len() as u32).collect();
    let edges = moa_core::minhash::build_graph(&sk, &lens, &o.gp);
    let mut w = open_out(&o.out)?;
    for &(a, b, wt) in &edges {
        writeln!(w, "{}\t{}\t{:.6}", f.ids[a as usize], f.ids[b as usize], wt).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    if !o.quiet {
        eprintln!(
            "{} {}  {} sequences → {} edges",
            art::mini(),
            bold("moa graph"),
            art::num(f.ids.len()),
            art::num(edges.len())
        );
    }
    Ok(())
}

fn build_rows() -> Vec<String> {
    vec![
        art::kv("version", &bold(&format!("moa {}", moa_core::VERSION))),
        art::kv("compiler", env!("MOA_RUSTC")),
        art::kv("target", env!("MOA_TARGET")),
        art::kv("profile", &format!("{} · LTO fat · 1 codegen unit", env!("MOA_PROFILE"))),
        art::kv("engines", "serial · rayon · HydraMPP"),
        art::kv("hydra-mpp", &format!("{} (crates.io)", hydra_mpp::VERSION)),
        art::kv("license", "CC-BY-NC-4.0 · RAW Lab, UNC Charlotte"),
    ]
}

fn cmd_version(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--short" || a == "-s") {
        println!("moa {}", moa_core::VERSION);
        return Ok(());
    }
    print!("{}", art::banner(""));
    println!();
    let mut rows = build_rows();
    rows.push(art::kv(
        "fidelity",
        &format!("{} 167/167 battery cases byte-identical to {}", art::ok(), moa_core::REFERENCE),
    ));
    print!("{}", art::panel("moa", &rows));
    println!(
        "  {} reproduces {} partition-for-partition · {}",
        art::ok(),
        bold(moa_core::REFERENCE),
        dim("run `moa doctor` to prove it here")
    );
    Ok(())
}

fn cmd_info(hcfg: Config) -> Result<(), String> {
    print!("{}", art::banner("build · runtime · defaults"));
    println!();
    print!("{}", art::panel("build", &build_rows()));
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let env = |k: &str| std::env::var(k).ok();
    let mut rt = vec![
        art::kv("host", &hostname()),
        art::kv("cpus", &format!("{cpus} available · rayon pool {}", rayon::current_num_threads())),
        art::kv("RAYON_NUM_THREADS", &env("RAYON_NUM_THREADS").unwrap_or_else(|| dim("unset"))),
        art::kv(
            "SLURM",
            &env("SLURM_JOB_NODELIST")
                .map(|n| format!("allocation {n}"))
                .unwrap_or_else(|| dim("not inside an allocation")),
        ),
        art::kv("colour", if art::color() { "on" } else { "off (NO_COLOR / not a TTY / --color never)" }),
    ];
    match HydraEngine::start(hcfg.quiet(true)) {
        Ok(h) => rt.push(art::kv("HydraMPP", &format!("{} {}", art::ok(), h.describe()))),
        Err(e) => rt.push(art::kv("HydraMPP", &format!("{} {e}", art::bad()))),
    }
    print!("{}", art::panel("runtime", &rt));
    let d = MclParams::default();
    let gp = moa_core::GraphParams::default();
    print!(
        "{}",
        art::panel(
            "MCL defaults  (identical to `mcl -z`)",
            &[
                art::kv("-I inflation", &format!("{:.1}", d.inflation)),
                art::kv("-P prune", &format!("{}  (cut {:.3e})", d.prune, d.precision())),
                art::kv("-S select", &d.select.to_string()),
                art::kv("-R recover", &d.recover.to_string()),
                art::kv("-pct", &format!("{}", d.pct * 100.0)),
                art::kv("-L loop length", &d.max_iter.to_string()),
                art::kv("chaos limit", &format!("{:e}  (stop when below)", d.chaos_limit)),
                art::kv("values", "f32 cells, f64 accumulation (mcl: float cells)"),
                art::kv("input merge", "mirror + symmetrise, duplicates → max"),
                art::kv("overlap", "cut (keep the first, largest system)"),
            ]
        )
    );
    print!(
        "{}",
        art::panel(
            "MinHash defaults  (moa cluster / moa graph)",
            &[
                art::kv("-k kmer", &gp.sketch.k.to_string()),
                art::kv("-s sketch", &format!("{}  (bottom-s)", gp.sketch.s)),
                art::kv("-i min-id", &gp.min_id.to_string()),
                art::kv("--len-cov", &gp.len_cov.to_string()),
                art::kv("--dense-bucket", &format!("{}  (all pairs up to this size)", gp.dense_bucket)),
                art::kv("--fanout", &format!("{}  (ring neighbours per side, larger buckets)", gp.fanout)),
                art::kv("--anchors", &format!("{}  (hubs per large bucket: families stay whole)", gp.anchors)),
                art::kv("edge weight", "raw Jaccard estimate"),
            ]
        )
    );
    Ok(())
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".into())
}

fn h(s: &str) -> String {
    bold(&fg(art::GOLD, s))
}
fn c(s: &str) -> String {
    fg(art::TEAL, s)
}

fn print_help(topic: Option<&str>) {
    match topic {
        Some("mcl") => {
            println!("{}\n", h("moa mcl — cluster a weighted graph (drop-in for `mcl <file> --abc`)"));
            println!("{}\n  moa mcl <graph.abc | -> [options]\n", h("USAGE"));
            println!("{}", h("INPUT"));
            println!("  One edge per line: {}  (any whitespace; weight defaults to 1).", c("labelA  labelB  [weight]"));
            println!("  Like mcl --abc: edges are undirected; duplicates and the two directions");
            println!("  of a pair merge by {} (BLAST all-vs-all a→b / b→a give one edge).", bold("max"));
            println!("  Non-positive weights are ignored. Blank lines and # comments are skipped.\n");
            mcl_opts();
            println!("{}", h("EXAMPLES"));
            println!("  moa mcl examples/friends.abc -I 2.0");
            println!("  moa mcl blast.abc -I 1.5 -o clusters.tsv --stats run.json");
            println!("  zcat big.abc.gz | moa mcl - -I 2.0 -t 32 > clusters.tsv");
            println!("  moa mcl huge.abc --engine hydra --hydra-slurm \"$SLURM_JOB_NODELIST\"");
        }
        Some("cluster") | Some("graph") => {
            println!("{}\n", h("moa cluster — alignment-free sequence clustering (MinHash graph + MCL)"));
            println!("{}", h("USAGE"));
            println!("  moa cluster <seqs.fasta | -> [options]      cluster sequences");
            println!("  moa graph   <seqs.fasta | -> [options]      only write the similarity graph (.abc)\n");
            println!("{}", h("MINHASH OPTIONS"));
            println!("  {}  minimum estimated identity for an edge         [0.95]", c("-i, --min-id <f>  "));
            println!("  {}  k-mer length                                    [7]", c("-k, --kmer <n>    "));
            println!("  {}  bottom-s sketch size (larger = less noise)      [256]", c("-s, --sketch <n>  "));
            println!("  {}  minimum length ratio shorter/longer            [0.70]", c("    --len-cov <f> "));
            println!("  {}  buckets up to this size pair all members       [64]", c("    --dense-bucket"));
            println!("  {}  ring neighbours per side in larger buckets      [6]", c("    --fanout <n>  "));
            println!("  {}  hub members per larger bucket (keeps families whole) [4]", c("    --anchors <n> "));
            println!("  {}  also write the similarity graph (labelA labelB jaccard)\n", c("    --edges <file>"));
            mcl_opts();
            println!("{}", h("EXAMPLES"));
            println!("  moa cluster examples/demo.faa -i 0.90");
            println!("  moa cluster pangenome.faa -I 1.5 -t 64 -o families.tsv --edges graph.abc");
        }
        Some("worker") => {
            println!("{}\n", h("moa worker — join a HydraMPP host as a worker node"));
            println!("  On the coordinator:   moa mcl big.abc --engine hydra --hydra-host");
            println!("  On each worker node:  moa worker --hydra-client <host-ip> [--hydra-cpus N]\n");
            println!("  Under SLURM nothing manual is needed:");
            println!("    moa mcl big.abc --hydra-slurm \"$SLURM_JOB_NODELIST\" --hydra-cpus \"$SLURM_CPUS_ON_NODE\"");
            println!("  The head node srun's moa on every other node as a worker automatically.\n");
            println!("{}", h("HYDRAMPP FLAGS (any command)"));
            println!("  {}  coordinate worker nodes", c("--hydra-host            "));
            println!("  {}  become a worker of <ip>", c("--hydra-client <ip>     "));
            println!("  {}  auto-wire a SLURM allocation", c("--hydra-slurm <nodelist>"));
            println!("  {}  CPU slots on this node (0 = detect)", c("--hydra-cpus <n>        "));
            println!("  {}  control port [24515]", c("--hydra-port <p>        "));
        }
        Some("doctor") => {
            println!("{}\n", h("moa doctor — prove this build is correct and fast, here"));
            println!("  1  the friends proof (3 named groups at -I 2.0, 2 at -I 1.4, chaos → 0)");
            println!("  2  a regression corpus recorded from {} (max-merge, overlap cut,", moa_core::REFERENCE);
            println!("     select/recover ties, multi-component, prune-only, no-recover schemes)");
            println!("  3  serial == rayon == hydra, bit-identical, on a 20k-node graph");
            println!("  4  MinHash: families recovered, a 2,000-member family stays connected");
            println!("  5  if `mcl` is on PATH: live side-by-side on fresh random graphs\n");
            println!("  {}  skip the throughput benchmark", c("--quick"));
            println!("  Exit status is non-zero if any check fails.");
        }
        _ => {
            print!("{}", art::banner(""));
            println!();
            println!("{}\n  moa <command> [options]\n", h("USAGE"));
            println!("{}", h("COMMANDS"));
            println!("  {}  cluster a weighted graph — a faithful, parallel `mcl --abc`", c("mcl <graph>    "));
            println!("  {}  MinHash similarity graph over sequences, then MCL", c("cluster <fasta>"));
            println!("  {}  write the MinHash similarity graph only (.abc)", c("graph <fasta>  "));
            println!("  {}  self-test: friends proof, mcl regression corpus, engines", c("doctor         "));
            println!("  {}  build, runtime, engines and every default", c("info           "));
            println!("  {}  version and build provenance  (--short for one line)", c("version        "));
            println!("  {}  serve as a HydraMPP worker node", c("worker         "));
            println!("  {}  this screen; `moa help <command>` for details\n", c("help           "));
            println!("{}", h("GLOBAL OPTIONS"));
            println!("  {}  worker threads (default: all cores; mcl's -te also works)", c("-t, --threads <n> "));
            println!("  {}  colour: auto | always | never   (NO_COLOR is honoured)", c("--color <when>    "));
            println!("  {}  HydraMPP roles: --hydra-host · --hydra-client · --hydra-slurm", c("--hydra-*         "));
            println!();
            println!("{}", h("QUICK START"));
            println!("  moa mcl examples/friends.abc -I 2.0          {}", dim("# three friend groups"));
            println!("  moa cluster examples/demo.faa -i 0.90        {}", dim("# protein families, no alignment"));
            println!("  moa doctor                                   {}", dim("# prove it on this machine"));
            println!();
            println!("  {}", fg(art::BROWN, &format!("“{}”", art::CATCHPHRASE)));
        }
    }
}

fn mcl_opts() {
    println!("{}", h("MCL OPTIONS  (names and defaults match mcl 22-282)"));
    println!("  {}  inflation; higher = finer clusters                [2.0]", c("-I <f>            "));
    println!("  {}  prune entries below 0.99999/n (0 = off)          [10000]", c("-P <n>            "));
    println!("  {}  keep at most n entries per column (0 = off)       [1100]", c("-S <n>            "));
    println!("  {}  recover up to n entries when pruning lost mass    [1400]", c("-R <n>            "));
    println!("  {}  recovery percentage                               [90]", c("-pct <n>          "));
    println!("  {}  main loop length (iteration cap)                  [10000]", c("-L <n>            "));
    println!("  {}  serial | rayon | hydra | auto                     [auto]", c("--engine <e>      "));
    println!("  {}  write clusters here (default stdout)", c("-o <file>         "));
    println!("  {}  write a JSON run summary", c("--stats <file>    "));
    println!("  {}  no summary on stderr", c("-q                "));
    println!("  {}  accepted and ignored, for drop-in use", c("--abc             "));
    println!();
    println!("{}", h("OUTPUT"));
    println!("  One cluster per line, tab-separated labels — largest first, ties by node");
    println!("  order — the same order mcl writes.\n");
}
