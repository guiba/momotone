use midir::{MidiOutput, MidiOutputPort};

fn main() {
    let out = MidiOutput::new("test").unwrap();
    let ports = out.ports();
    for port in ports {
        println!("Port: {:?}", out.port_name(&port));
    }
}
