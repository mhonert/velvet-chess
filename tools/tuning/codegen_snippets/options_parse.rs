
pub fn parse_set_option(tx: &Sender<Message>, name: &str, value_str: &str) {
    if SINGLE_VALUE_OPTION_NAMES.contains(&name) {
        set_option_value(tx, name, value_str);
        return;
    }

    let name_without_index = name.replace(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][..], "");
    if MULTI_VALUE_OPTION_NAMES.contains(&name_without_index.as_str()) {
        set_array_option_value(tx, name_without_index.as_str(), name, value_str);
        return;
    }
}

fn set_option_value(tx: &Sender<Message>, name: &str, value_str: &str) {
    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetOption(String::from(name), value));
}

fn set_array_option_value(tx: &Sender<Message>, name: &str, name_with_index: &str, value_str: &str) {
    let index = match i32::from_str(&name_with_index[name.len()..]) {
        Ok(index) => index,
        Err(_) => {
            println!("Invalid index: {}", name_with_index);
            return;
        }
    };

    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetArrayOption(String::from(name), index, value));
}

fn send_message(tx: &Sender<Message>, msg: Message) {
    match tx.send(msg) {
        Ok(_) => return,
        Err(err) => {
            eprintln!("could not send message to engine thread: {}", err);
        }
    }
}
