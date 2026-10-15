//! `moa doctor` — prove the build is correct (and quick) on this machine.

use hydra_mpp::Config;
use moa_core::art::{self, bold, dim, fg};
use moa_core::engine::{Engine, HydraEngine};
use moa_core::regress::CASES;
use moa_core::synth::{family, synthetic};
use moa_core::{cluster_abc, Graph, GraphParams, MclParams};
use std::collections::BTreeSet;
use std::time::Instant;

type Partition = BTreeSet<BTreeSet<String>>;

fn partition(labels: &[String], clusters: &[Vec<u32>]) -> Partition {
    clusters.iter().map(|c| c.iter().map(|&m| labels[m as usize].clone()).collect()).collect()
}

struct Tally {
    pass: usize,
    fail: usize,
}
impl Tally {
    fn check(&mut self, ok: bool, what: &str, detail: &str) {
        if ok {
            self.pass += 1;
            println!("  {} {}  {}", art::ok(), art::pad(what, 46), dim(detail));
        } else {
            self.fail += 1;
            println!("  {} {}  {}", art::bad(), art::pad(&bold(what), 46), fg(art::RED, detail));
        }
    }
}

fn section(n: usize, title: &str) {
    println!("\n{} {}", bold(&fg(art::GOLD, &format!("{n} ▸"))), bold(title));
}

pub fn run(args: &[String], hcfg: Config) -> Result<(), String> {
    let quick = args.iter().any(|a| a == "--quick");
    print!("{}", art::banner("doctor · proving this build against the reference, on this machine"));
    let mut t = Tally { pass: 0, fail: 0 };
    let t_all = Instant::now();

    // 1 ─ friends ----------------------------------------------------------------
    section(1, "The friends proof  (van Dongen's canonical example)");
    let d = MclParams::default();
    let (lab, out) = cluster_abc(moa_core::FRIENDS_ABC.as_bytes(), &d, &Engine::Serial)?;
    let got = partition(&lab, &out.clusters);
    let want: Partition = [
        &["ann", "fred", "gilian", "john"][..],
        &["bob", "charlie", "esme"][..],
        &["dan", "harriet", "indy", "kelly", "lucy"][..],
    ]
    .iter()
    .map(|c| c.iter().map(|s| s.to_string()).collect())
    .collect();
    t.check(got == want, "-I 2.0 → {ann fred john gilian} {bob charlie esme}", "{dan harriet indy kelly lucy}");
    t.check(
        out.chaos < d.chaos_limit && out.converged,
        "chaos falls below 1e-4  (mcl's stop rule)",
        &format!("{} iterations, final chaos {:.1e}", out.iterations, out.chaos),
    );
    let p14 = MclParams { inflation: 1.4, ..d };
    let (_, o14) = cluster_abc(moa_core::FRIENDS_ABC.as_bytes(), &p14, &Engine::Serial)?;
    t.check(o14.clusters.len() == 2, "-I 1.4 coarsens to 2 groups", &format!("{} clusters", o14.clusters.len()));

    // 2 ─ regression corpus --------------------------------------------------------
    section(2, &format!("Regression corpus recorded from {}", moa_core::REFERENCE));
    for c in CASES {
        let p = MclParams { inflation: c.inflation, prune: c.prune, select: c.select, recover: c.recover, ..d };
        let (lab, out) = cluster_abc(c.abc.as_bytes(), &p, &Engine::Serial)?;
        let got = partition(&lab, &out.clusters);
        let want: Partition = c.expected.iter().map(|cl| cl.iter().map(|s| s.to_string()).collect()).collect();
        let old = match (c.old_1_0_0_ok, c.old_1_1_0_ok) {
            (true, Some(true)) | (true, None) => String::new(),
            (a, b) => format!(
                " · was wrong in {}",
                [(!a).then_some("1.0.0"), b.and_then(|b| (!b).then_some("1.1.0"))]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(" & ")
            ),
        };
        let sch = if (c.prune, c.select, c.recover) == (10000, 1100, 1400) {
            String::new()
        } else {
            format!(" -P {} -S {} -R {}", c.prune, c.select, c.recover)
        };
        t.check(got == want, c.name, &format!("-I {}{sch} → {} clusters{old}", c.inflation, want.len()));
    }

    // 3 ─ engines ----------------------------------------------------------------------
    section(3, "Engines agree bit-for-bit  (serial · rayon · HydraMPP)");
    let (n, edges) = synthetic(0x5EED, if quick { 3000 } else { 12_000 }, if quick { 600 } else { 2500 });
    let g = Graph::from_edges(n, &edges);
    let mut engines: Vec<Engine> = vec![Engine::Serial, Engine::Rayon];
    match HydraEngine::start(hcfg.quiet(true)) {
        Ok(h) => engines.push(Engine::Hydra(h)),
        Err(e) => t.check(false, "HydraMPP runtime starts", &e),
    }
    let mut reference = None;
    let mut times = Vec::new();
    for eng in &engines {
        let t0 = Instant::now();
        let out = moa_core::mcl::run(g.clone(), &d, eng);
        let secs = t0.elapsed().as_secs_f64();
        let sig = (out.clusters.clone(), out.iterations, out.chaos.to_bits());
        let same = match &reference {
            None => {
                reference = Some(sig);
                true
            }
            Some(r) => *r == sig,
        };
        times.push(secs);
        t.check(
            same,
            &format!("{:<7} {}", eng.name(), art::num(out.clusters.len()) + " clusters"),
            &format!("{} · {} it · {} · {}", art::num(n) + " nodes", out.iterations, art::dur(secs), eng.describe()),
        );
    }
    if times.len() >= 2 && times[1] > 0.0 {
        println!(
            "  {}",
            dim(&format!(
                "rayon speed-up over serial: {:.2}× on {} threads",
                times[0] / times[1],
                rayon::current_num_threads()
            ))
        );
    }

    // 4 ─ MinHash -----------------------------------------------------------------------
    section(4, "MinHash front-end");
    let demo = include_str!("../examples/demo.faa");
    let gp = GraphParams { min_id: 0.90, ..GraphParams::default() };
    let r = moa_core::cluster_fasta(demo.as_bytes(), &gp, &d, &Engine::Rayon)?;
    let fam: Partition = partition(&r.ids, &r.out.clusters);
    let fam_ok = fam.len() == 4
        && fam.iter().all(|c| {
            let p: BTreeSet<&str> = c.iter().map(|s| s.split('_').next().unwrap()).collect();
            p.len() == 1
        });
    t.check(
        fam_ok,
        "demo.faa → 3 families + 1 singleton",
        &format!("{} sequences, {} edges", r.ids.len(), r.edges.len()),
    );
    // A single-copy core gene seen in N genomes must come out as ONE family —
    // connected is not enough (1.1.0's ring index kept it connected, and MCL
    // then shattered it into ~1,300 clusters).
    let fam_n = if quick { 400 } else { 1200 };
    let seqs = family(fam_n, 300, 7);
    let fasta: String =
        seqs.iter().enumerate().map(|(i, s)| format!(">core_{i}\n{}\n", String::from_utf8_lossy(s))).collect();
    let t0 = Instant::now();
    let fr = moa_core::cluster_fasta(fasta.as_bytes(), &GraphParams::default(), &d, &Engine::Rayon)?;
    t.check(
        fr.out.clusters.len() == 1,
        &format!("{}-member core gene → 1 family", art::num(fam_n)),
        &format!(
            "{} edges (not {}) · {} · {}",
            art::num(fr.edges.len()),
            art::num(fam_n * (fam_n - 1) / 2),
            if fr.out.clusters.len() == 1 {
                "1 cluster".to_string()
            } else {
                format!("{} clusters", fr.out.clusters.len())
            },
            art::dur(t0.elapsed().as_secs_f64())
        ),
    );

    // 5 ─ live reference -------------------------------------------------------------------
    section(5, "Live side-by-side with the reference binary");
    match std::process::Command::new("mcl").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let ver = String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("mcl").to_string();
            let dir = std::env::temp_dir().join(format!("moa-doctor-{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let mut ok = 0;
            let total = 8;
            for s in 0..total {
                let (n, e) = synthetic(1000 + s as u64, 150 + 40 * s, 30);
                let text: String = e.iter().map(|(a, b, w)| format!("n{a}\tn{b}\t{w}\n")).collect();
                let path = dir.join(format!("g{s}.abc"));
                std::fs::write(&path, &text).map_err(|e| e.to_string())?;
                let infl = [1.4, 2.0, 3.0, 5.0][s % 4];
                let ref_out = std::process::Command::new("mcl")
                    .arg(&path)
                    .args(["--abc", "-I", &infl.to_string(), "-o", "-"])
                    .output()
                    .map_err(|e| e.to_string())?;
                let want: Partition = String::from_utf8_lossy(&ref_out.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| l.split('\t').map(|s| s.to_string()).collect())
                    .collect();
                let p = MclParams { inflation: infl, ..d };
                let (lab, out) = cluster_abc(text.as_bytes(), &p, &Engine::Rayon)?;
                if partition(&lab, &out.clusters) == want {
                    ok += 1;
                }
                let _ = n;
            }
            let _ = std::fs::remove_dir_all(&dir);
            t.check(ok == total, &format!("{ok}/{total} fresh graphs identical to `mcl`"), &ver);
        }
        _ => println!(
            "  {} {}",
            art::skip(),
            dim("`mcl` not on PATH — install it (e.g. apt install mcl) for a live comparison")
        ),
    }

    // ─ verdict ---------------------------------------------------------------------------------
    println!();
    let total = t.pass + t.fail;
    if t.fail == 0 {
        println!(
            "  {}  {}  {}",
            art::mini(),
            bold(&fg(art::GREEN, &format!("ALL {total} CHECKS PASSED"))),
            dim(&format!("in {}", art::dur(t_all.elapsed().as_secs_f64())))
        );
        println!("  {}", fg(art::BROWN, &format!("“{}”", art::CATCHPHRASE)));
        Ok(())
    } else {
        Err(format!("{} of {total} checks FAILED", t.fail))
    }
}
