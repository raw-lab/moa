//! Banners, the moa mascot, and the catchphrase.
//!
//! The moa was a giant flightless bird of New Zealand — a ratite that gathered
//! in flocks. Fitting, for a tool whose whole job is to find the flocks hiding
//! in a graph.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The catchphrase.
pub const CATCHPHRASE: &str = "Extinct bird. Living clusters.";

/// The one-line description used under the wordmark.
pub const TAGLINE: &str = "Markov clustering + MinHash, forged 100% in Rust.";

/// The moa's face.
pub const MOA_FACE: &str = r#"              .-"-.
             ( o o )
              \ v /
               '-'"#;

/// The wordmark: "MOA" in bold block letters (FIGlet "ANSI Shadow").
pub const MOA_WORD: &str = r#"   ███╗   ███╗ ██████╗  █████╗
   ████╗ ████║██╔═══██╗██╔══██╗
   ██╔████╔██║██║   ██║███████║
   ██║╚██╔╝██║██║   ██║██╔══██║
   ██║ ╚═╝ ██║╚██████╔╝██║  ██║
   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝"#;

/// Face above the wordmark — the full mascot.
pub fn logo() -> String {
    format!("{MOA_FACE}\n{MOA_WORD}")
}

/// A compact single-line bird for inline notes and the footer.
pub const MOA_MINI: &str = "~<( o )>~";

/// Render the full banner: mascot + a version/tagline line + a subtitle line.
pub fn banner(subtitle: &str) -> String {
    format!(
        "{logo}\n   v{ver} · {tag}\n   \"{cat}\"\n   {sub}\n",
        logo = logo(),
        ver = VERSION,
        tag = TAGLINE,
        cat = CATCHPHRASE,
        sub = subtitle,
    )
}

/// A slimmer header (no mascot) for sub-reports like `doctor` sections.
pub fn rule(title: &str) -> String {
    let bar = "─".repeat(46);
    format!("{bar}\n  {title}\n{bar}")
}
