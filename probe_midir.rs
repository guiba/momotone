use midir::MidiOutput;

fn main() {
    let out = MidiOutput::new("test").unwrap();
    println!("{:?}", out);
}
