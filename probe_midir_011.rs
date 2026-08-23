use midir::MidiOutput;

fn main() {
    let out = MidiOutput::new("test").unwrap();
    // We want to see what methods are available on out.
    // Since we can't easily "print all methods", 
    // let's try some common ones or just see if it compiles with certain calls.
}

mod dummy {
    use midir::MidiOutput;
    pub fn check(out: &MidiOutput) {
        // Attempting to use potential method names from common patterns
        // _ = out.open("test", |_| {}); 
        // _ = out.open_port("test");
    }
}
