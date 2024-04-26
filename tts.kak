define-command tts_narrate_from_cursor %{
    evaluate-commands -draft %{
      execute-keys -draft -save-regs '' '%y';
      echo -to-file /tmp/kak_tts_fifo %sh{
          kak_reg_dquote=$(echo "$kak_reg_dquote" | sed 's/"/\\"/g')
          body="{\"method\":\"narrate_from_cursor\",\"params\":{\"buffer\": \"${kak_reg_dquote}\",\"cursor_byte_offset\":${kak_cursor_byte_offset}}}"
          echo "Content-Length: ${#body}\r\n\r\n${body}" }
    }
}


define-command tts_cancel %{
  echo -to-file /tmp/kak_tts_fifo %sh{
      body="{\"method\":\"cancel\"}"
      echo "Content-Length: ${#body}\r\n\r\n${body}" }
}
