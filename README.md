# Install

```
cd ~/.config/kak/plugins
git clone https://github.com/tom-huntington/tts.kak.git
pip install tts.kak
echo 'source %val{config}/plugins/tts.kak/tts.kak"' >> ~/.config/kak/kakrc
```

or with plug.kak (which I'm not using atm, so this might me wrong)
```
plug "tom-huntington/tts.kak" do %{
    pip install .
}
```

I'm using python 3.10.

# Usage

Start the server
```sh
tts_server
```

in another terminal
```sh
kak
:tts_narrate_from_cursor
:tts_cancel
```

`ctrl-c` wont work to stop the server.
The script wont exit because it is blocking, reading from the fifo.
To unblock simply `:tts_cancel` or `echo " " > /tmp/kak_tts_fifo`