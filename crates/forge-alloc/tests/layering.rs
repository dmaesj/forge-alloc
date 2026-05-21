//! Source-scan layering guard.
//!
//! `forge-alloc` bundles three internal layers — `backing` (L1), `layout`
//! (L2), `hardening` (L3) — that must form a strict bottom-up DAG: a lower
//! layer may never reference a higher one. Rust cannot enforce this *within*
//! a crate (modules may reference each other freely, even cyclically), so
//! this test fails the build if a forbidden cross-layer reference appears.
//!
//! The `forge-alloc-core` trait-layer boundary, by contrast, *is*
//! compiler-enforced — it is a separate crate, so nothing here re-checks it.

use std::fs;
use std::path::Path;

/// Walk every `.rs` file under `src/<lower>/` and assert that none of them
/// reference a higher layer via a `crate::<upper>` path. Comment lines are
/// skipped so prose mentions don't trip the guard.
fn assert_no_upward_refs(lower: &str, forbidden: &[&str]) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(lower);
    let mut violations = Vec::new();
    visit(&dir, &mut |path, contents| {
        for (idx, line) in contents.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for upper in forbidden {
                if line.contains(&format!("crate::{upper}")) {
                    violations.push(format!(
                        "{}:{}: `{lower}` layer references `{upper}` — {}",
                        path.display(),
                        idx + 1,
                        line.trim(),
                    ));
                }
            }
        }
    });
    assert!(
        violations.is_empty(),
        "layering violation(s) — a lower layer must not depend on a higher one:\n{}",
        violations.join("\n"),
    );
}

fn visit(dir: &Path, f: &mut dyn FnMut(&Path, &str)) {
    for entry in fs::read_dir(dir).expect("read module directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit(&path, f);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let contents = fs::read_to_string(&path).expect("read source file");
            f(&path, &contents);
        }
    }
}

#[test]
fn backing_does_not_reference_higher_layers() {
    assert_no_upward_refs("backing", &["layout", "hardening"]);
}

#[test]
fn layout_does_not_reference_hardening() {
    assert_no_upward_refs("layout", &["hardening"]);
}
