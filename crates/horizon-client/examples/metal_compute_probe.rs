//! Compiles the game's compute shaders, and a few tiny ones, on this machine's GPU and says which
//! compile. For finding what an older GPU driver's shader compiler (Metal on an old Intel GPU, say)
//! chokes on: `cargo run --release -p horizon-client --example metal_compute_probe`.

/// Does nothing: if this fails, no compute shader will compile here.
const EMPTY: &str = "
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {}
";

/// Storage buffers and the integer maths of `animate.wgsl`, no texture.
const BUFFERS: &str = "
struct Job { layer: u32, current: u32, next: u32, millis: u32, edge: u32 }
@group(0) @binding(0) var<storage, read> frames: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let job = jobs[id.z];
    let size = 8u;
    var texel: u32;
    if size >= job.edge {
        let scale = size / job.edge;
        texel = id.y / scale * job.edge + id.x / scale;
    } else {
        texel = 4u * (job.edge * job.edge - size * size) / 3u + id.y * size + id.x;
    }
    let a = frames[job.current + texel];
    let b = frames[job.next + texel];
    out[id.x] = (a * (1000u - job.millis) + b * job.millis + 500u) / 1000u;
}
";

/// A write-only 2D storage texture.
const TEXTURE_2D: &str = "
@group(0) @binding(0) var level: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(level, id.xy, vec4<f32>(1.0));
}
";

/// A write-only 2D array storage texture, as `animate.wgsl` writes.
const TEXTURE_ARRAY: &str = "
@group(0) @binding(0) var level: texture_storage_2d_array<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(level, id.xy, 0u, vec4<f32>(1.0));
}
";

/// The same, with `textureDimensions` and an early return, as `animate.wgsl` has.
const TEXTURE_ARRAY_DIMENSIONS: &str = "
@group(0) @binding(0) var level: texture_storage_2d_array<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(level).x;
    if id.x >= size || id.y >= size {
        return;
    }
    textureStore(level, id.xy, id.z, vec4<f32>(1.0));
}
";

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("no GPU adapter");
    println!("adapter: {:?}", adapter.get_info());
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Compute probe"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");

    let probe = |name: &str, source: &str, entry: &str| {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        // Native error scopes resolve at once.
        match pollster::block_on(scope.pop()) {
            None => println!("ok      {name} [{entry}]"),
            Some(error) => println!("FAILED  {name} [{entry}]: {error}"),
        }
    };

    probe("empty kernel", EMPTY, "main");
    probe("storage buffers and integer maths", BUFFERS, "main");
    probe("write-only 2D storage texture", TEXTURE_2D, "main");
    probe("write-only 2D array storage texture", TEXTURE_ARRAY, "main");
    probe(
        "2D array storage texture with textureDimensions",
        TEXTURE_ARRAY_DIMENSIONS,
        "main",
    );
    probe(
        "game: animate.wgsl",
        include_str!("../src/gfx/animate.wgsl"),
        "animate",
    );
    probe(
        "game: flatten.wgsl",
        include_str!("../src/gfx/flatten.wgsl"),
        "flatten",
    );
    probe(
        "game: flatten.wgsl",
        include_str!("../src/gfx/flatten.wgsl"),
        "flatten_seen",
    );
    println!("done");
}
