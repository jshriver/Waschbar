// Prints the best build this CPU can run: x86-64-v1..v4 or armv8.
// Used by the release workflow to decide which builds a runner can test.
fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let v2 = has!("popcnt") && has!("sse3") && has!("ssse3") && has!("sse4.1") && has!("sse4.2")
            && has!("cmpxchg16b");
        let v3 = v2 && has!("avx") && has!("avx2") && has!("bmi1") && has!("bmi2") && has!("fma")
            && has!("lzcnt") && has!("movbe") && has!("f16c");
        let v4 = v3 && has!("avx512f") && has!("avx512bw") && has!("avx512cd") && has!("avx512dq")
            && has!("avx512vl");
        println!("x86-64-v{}", if v4 { 4 } else if v3 { 3 } else if v2 { 2 } else { 1 });
    }
    #[cfg(target_arch = "aarch64")]
    println!("armv8");
}
