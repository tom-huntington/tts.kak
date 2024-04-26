define-command tts_narrate_from_cursor %{
    evaluate-commands -draft %{
      execute-keys -draft -save-regs '' '%y';
      echo -to-file /tmp/kak_tts_fifo %sh{
          body="{\"method\":\"narrate_from_cursor\",\"params\":{\"buffer\": \"${kak_reg_dquote}\",\"cursor_byte_offset\":${kak_cursor_byte_offset}}}"
          echo "Content-Length: ${#body}\r\n\r\n${body}" }
    }
}

define-command tts_write_to_file %{
    evaluate-commands -draft %{
      execute-keys '%';
      echo -to-file /tmp/kak_tts_fifo %sh{
          body="{\"method\":\"write_to_file\",\"params\":{\"buffer\": \"${kak_selection}\",\"bufname\":\"${kak_bufname}\"}}"
          echo "Content-Length: ${#body}\r\n\r\n${body}" }
    }
}


define-command tts_cancel %{
  echo -to-file /tmp/kak_tts_fifo %sh{
      body="{\"method\":\"cancel\"}"
      echo "Content-Length: ${#body}\r\n\r\n${body}" }
}
