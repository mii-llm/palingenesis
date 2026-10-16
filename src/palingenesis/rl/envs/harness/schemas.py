"""Parameter schemas copied verbatim from requests the real harnesses sent to a recording stub (pgs harness serve):
kimi-cli 1.52.0, qwen-code 0.25.2 (local runs) and opencode 1.19.2 (Harbor 0.24 CLI run) (research/tiny-reliable-agent/harness/fixtures/). Interface contracts: keep them
byte-exact; tool descriptions are paraphrased in profiles_more.py."""

# fmt: off
KIMI_PARAMETERS = {'Shell': {'properties': {'command': {'description': 'The command to execute.', 'type': 'string'},
                          'timeout': {'default': 60,
                                      'description': 'The timeout in seconds for the command to execute. If '
                                                     'the command takes longer than this, it will be killed.',
                                      'maximum': 86400,
                                      'minimum': 1,
                                      'type': 'integer'},
                          'run_in_background': {'default': False,
                                                'description': 'Whether to run the command as a background '
                                                               'task.',
                                                'type': 'boolean'},
                          'description': {'default': '',
                                          'description': 'A short description for the background task. '
                                                         'Required when run_in_background=true.',
                                          'type': 'string'}},
           'required': ['command'],
           'type': 'object'},
 'ReadFile': {'properties': {'path': {'description': 'The path to the file to read. Absolute paths are '
                                                     'required when reading files outside the working '
                                                     'directory.',
                                      'type': 'string'},
                             'line_offset': {'default': 1,
                                             'description': 'The line number to start reading from. By default '
                                                            'read from the beginning of the file. Set this '
                                                            'when the file is too large to read at once. '
                                                            'Negative values read from the end of the file '
                                                            '(e.g. -100 reads the last 100 lines). The '
                                                            'absolute value of negative offset cannot exceed '
                                                            '1000.',
                                             'type': 'integer'},
                             'n_lines': {'default': 1000,
                                         'description': 'The number of lines to read. By default read up to '
                                                        '1000 lines, which is the max allowed value. Set this '
                                                        'value when the file is too large to read at once.',
                                         'minimum': 1,
                                         'type': 'integer'}},
              'required': ['path'],
              'type': 'object'},
 'Glob': {'properties': {'pattern': {'description': 'Glob pattern to match files/directories.',
                                     'type': 'string'},
                         'directory': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                       'default': None,
                                       'description': 'Absolute path to the directory to search in (defaults '
                                                      'to working directory).'},
                         'include_dirs': {'default': True,
                                          'description': 'Whether to include directories in results.',
                                          'type': 'boolean'}},
          'required': ['pattern'],
          'type': 'object'},
 'Grep': {'properties': {'pattern': {'description': 'The regular expression pattern to search for in file '
                                                    'contents',
                                     'type': 'string'},
                         'path': {'default': '.',
                                  'description': 'File or directory to search in. Defaults to current working '
                                                 'directory. If specified, it must be an absolute path.',
                                  'type': 'string'},
                         'glob': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                  'default': None,
                                  'description': 'Glob pattern to filter files (e.g. `*.js`, `*.{ts,tsx}`). No '
                                                 'filter by default.'},
                         'output_mode': {'default': 'files_with_matches',
                                         'description': '`content`: Show matching lines (supports `-B`, `-A`, '
                                                        '`-C`, `-n`, `head_limit`); `files_with_matches`: Show '
                                                        'file paths (supports `head_limit`); `count_matches`: '
                                                        'Show total number of matches. Defaults to '
                                                        '`files_with_matches`.',
                                         'type': 'string'},
                         '-B': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                                'default': None,
                                'description': 'Number of lines to show before each match (the `-B` option). '
                                               'Requires `output_mode` to be `content`.'},
                         '-A': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                                'default': None,
                                'description': 'Number of lines to show after each match (the `-A` option). '
                                               'Requires `output_mode` to be `content`.'},
                         '-C': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                                'default': None,
                                'description': 'Number of lines to show before and after each match (the `-C` '
                                               'option). Requires `output_mode` to be `content`.'},
                         '-n': {'default': True,
                                'description': 'Show line numbers in output (the `-n` option). Requires '
                                               '`output_mode` to be `content`. Defaults to true.',
                                'type': 'boolean'},
                         '-i': {'default': False,
                                'description': 'Case insensitive search (the `-i` option).',
                                'type': 'boolean'},
                         'type': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                  'default': None,
                                  'description': 'File type to search. Examples: py, rust, js, ts, go, java, '
                                                 'etc. More efficient than `glob` for standard file types.'},
                         'head_limit': {'anyOf': [{'minimum': 0, 'type': 'integer'}, {'type': 'null'}],
                                        'default': 250,
                                        'description': 'Limit output to first N lines/entries, equivalent to '
                                                       '`| head -N`. Works across all output modes: content '
                                                       '(limits output lines), files_with_matches (limits file '
                                                       'paths), count_matches (limits count entries). Defaults '
                                                       'to 250. Pass 0 for unlimited (use sparingly — large '
                                                       'result sets waste context).'},
                         'offset': {'default': 0,
                                    'description': 'Skip first N lines/entries before applying head_limit, '
                                                   'equivalent to `| tail -n +N | head -N`. Works across all '
                                                   'output modes. Defaults to 0.',
                                    'minimum': 0,
                                    'type': 'integer'},
                         'multiline': {'default': False,
                                       'description': 'Enable multiline mode where `.` matches newlines and '
                                                      'patterns can span lines (the `-U` and '
                                                      '`--multiline-dotall` options). By default, multiline '
                                                      'mode is disabled.',
                                       'type': 'boolean'},
                         'include_ignored': {'default': False,
                                             'description': 'Include files that are ignored by `.gitignore`, '
                                                            '`.ignore`, and other ignore rules. Useful for '
                                                            'searching gitignored artifacts such as build '
                                                            'outputs (e.g. `dist/`, `build/`) or '
                                                            '`node_modules`. Sensitive files (like `.env`) '
                                                            'remain filtered by the sensitive-file protection '
                                                            'layer. Defaults to false.',
                                             'type': 'boolean'}},
          'required': ['pattern'],
          'type': 'object'},
 'WriteFile': {'properties': {'path': {'description': 'The path to the file to write. Absolute paths are '
                                                      'required when writing files outside the working '
                                                      'directory.',
                                       'type': 'string'},
                              'content': {'description': 'The content to write to the file', 'type': 'string'},
                              'mode': {'default': 'overwrite',
                                       'description': 'The mode to use to write to the file. Two modes are '
                                                      'supported: `overwrite` for overwriting the whole file '
                                                      'and `append` for appending to the end of an existing '
                                                      'file.',
                                       'enum': ['overwrite', 'append'],
                                       'type': 'string'}},
               'required': ['path', 'content'],
               'type': 'object'},
 'StrReplaceFile': {'properties': {'path': {'description': 'The path to the file to edit. Absolute paths are '
                                                           'required when editing files outside the working '
                                                           'directory.',
                                            'type': 'string'},
                                   'edit': {'anyOf': [{'properties': {'old': {'description': 'The old string '
                                                                                             'to replace. Can '
                                                                                             'be multi-line.',
                                                                              'type': 'string'},
                                                                      'new': {'description': 'The new string '
                                                                                             'to replace with. '
                                                                                             'Can be '
                                                                                             'multi-line.',
                                                                              'type': 'string'},
                                                                      'replace_all': {'default': False,
                                                                                      'description': 'Whether '
                                                                                                     'to '
                                                                                                     'replace '
                                                                                                     'all '
                                                                                                     'occurrences.',
                                                                                      'type': 'boolean'}},
                                                       'required': ['old', 'new'],
                                                       'type': 'object'},
                                                      {'items': {'properties': {'old': {'description': 'The '
                                                                                                       'old '
                                                                                                       'string '
                                                                                                       'to '
                                                                                                       'replace. '
                                                                                                       'Can be '
                                                                                                       'multi-line.',
                                                                                        'type': 'string'},
                                                                                'new': {'description': 'The '
                                                                                                       'new '
                                                                                                       'string '
                                                                                                       'to '
                                                                                                       'replace '
                                                                                                       'with. '
                                                                                                       'Can be '
                                                                                                       'multi-line.',
                                                                                        'type': 'string'},
                                                                                'replace_all': {'default': False,
                                                                                                'description': 'Whether '
                                                                                                               'to '
                                                                                                               'replace '
                                                                                                               'all '
                                                                                                               'occurrences.',
                                                                                                'type': 'boolean'}},
                                                                 'required': ['old', 'new'],
                                                                 'type': 'object'},
                                                       'type': 'array'}],
                                            'description': 'The edit(s) to apply to the file. You can provide '
                                                           'a single edit or a list of edits here.'}},
                    'required': ['path', 'edit'],
                    'type': 'object'}}

QWEN_CODE_PARAMETERS = {'edit': {'properties': {'file_path': {'description': 'The absolute path to the file to modify. Must start '
                                                      "with '/'.",
                                       'type': 'string'},
                         'old_string': {'description': 'The exact literal text to replace, preferably '
                                                       'unescaped. For single replacements (default), include '
                                                       'at least 3 lines of context BEFORE and AFTER the '
                                                       'target text, matching whitespace and indentation '
                                                       'precisely. If this string is not the exact literal '
                                                       'text (i.e. you escaped it) or does not match exactly, '
                                                       'the tool will fail.',
                                        'type': 'string'},
                         'new_string': {'description': 'The exact literal text to replace `old_string` with, '
                                                       'preferably unescaped. Provide the EXACT text. Ensure '
                                                       'the resulting code is correct and idiomatic.',
                                        'type': 'string'},
                         'replace_all': {'type': 'boolean',
                                         'description': 'Replace all occurrences of old_string (default '
                                                        'false).'}},
          'required': ['file_path', 'old_string', 'new_string'],
          'type': 'object'},
 'glob': {'properties': {'pattern': {'description': 'The glob pattern to match files against',
                                     'type': 'string'},
                         'path': {'description': 'The directory to search in. If not specified, the current '
                                                 'working directory will be used. IMPORTANT: Omit this field '
                                                 'to use the default directory. DO NOT enter "undefined" or '
                                                 '"null" - simply omit it for the default behavior. Must be a '
                                                 'valid directory path if provided.',
                                  'type': 'string'}},
          'required': ['pattern'],
          'type': 'object'},
 'grep_search': {'properties': {'pattern': {'type': 'string',
                                            'description': 'The regular expression pattern to search for in '
                                                           'file contents'},
                                'glob': {'type': 'string',
                                         'description': 'Glob pattern to filter files (e.g. "*.js", '
                                                        '"*.{ts,tsx}") - maps to rg --glob'},
                                'path': {'type': 'string',
                                         'description': 'File or directory to search in (rg PATH). Defaults to '
                                                        'current working directory.'},
                                'limit': {'type': 'integer',
                                          'minimum': 1,
                                          'description': 'Limit output to first N lines/entries. Must be a '
                                                         'positive integer. Optional - shows all matches if '
                                                         'not specified.'}},
                 'required': ['pattern'],
                 'type': 'object'},
 'read_file': {'properties': {'file_path': {'description': 'The absolute path to the file to read (e.g., '
                                                           "'/home/user/project/file.txt'). Relative paths are "
                                                           'not supported. You must provide an absolute path.',
                                            'type': 'string'},
                              'offset': {'description': 'Optional: For text files, the 0-based line number to '
                                                        "start reading from. Requires 'limit' to be set. Use "
                                                        'for paginating through large files. Omit or set to '
                                                        'null for Jupyter notebooks (.ipynb); null is treated '
                                                        'as omitted.',
                                         'type': ['integer', 'null']},
                              'limit': {'description': 'Optional: For text files, maximum number of lines to '
                                                       "read. Use with 'offset' to paginate through large "
                                                       'files. If omitted, reads the entire file (if feasible, '
                                                       'up to a default limit). Omit or set to null for '
                                                       'Jupyter notebooks (.ipynb); null is treated as '
                                                       'omitted.',
                                        'type': ['integer', 'null']},
                              'pages': {'description': 'Optional: For PDF files, the page range to extract as '
                                                       "text (e.g., '1-5', '3', '10-20'). Pages are 1-indexed. "
                                                       "Max 20 pages per request. Open-ended ranges like '3-' "
                                                       'are not supported. Use this for large PDFs or when the '
                                                       'model does not support native PDF input. Omit or set '
                                                       'to null for Jupyter notebooks (.ipynb); null is '
                                                       'treated as omitted.',
                                        'type': ['string', 'null']}},
               'required': ['file_path'],
               'type': 'object'},
 'run_shell_command': {'type': 'object',
                       'properties': {'command': {'type': 'string',
                                                  'description': 'Exact bash command to execute as `bash -c '
                                                                 '<command>`'},
                                      'is_background': {'type': 'boolean',
                                                        'description': 'Optional: Whether to run the command '
                                                                       'in background. If not specified, '
                                                                       'defaults to false (foreground '
                                                                       'execution). Explicitly set to true for '
                                                                       'long-running processes like '
                                                                       'development servers, watchers, or '
                                                                       'daemons that should continue running '
                                                                       'without blocking further commands.'},
                                      'timeout': {'type': 'integer',
                                                  'minimum': 1,
                                                  'maximum': 600000,
                                                  'description': 'Optional timeout in milliseconds (max '
                                                                 '600000)'},
                                      'description': {'type': 'string',
                                                      'description': 'Brief description of the command for the '
                                                                     'user. Be specific and concise. Ideally a '
                                                                     'single sentence. Can be up to 3 '
                                                                     'sentences for clarity. No line breaks.'},
                                      'directory': {'type': 'string',
                                                    'description': '(OPTIONAL) The absolute path of the '
                                                                   'directory to run the command in. If not '
                                                                   'provided, the project root directory is '
                                                                   'used. Must already exist. A directory '
                                                                   'outside the workspace requires approval.'}},
                       'required': ['command']},
 'write_file': {'properties': {'file_path': {'description': 'The absolute path to the file to write to (e.g., '
                                                            "'/home/user/project/file.txt'). Relative paths "
                                                            'are not supported.',
                                             'type': 'string'},
                               'content': {'description': 'The content to write to the file.',
                                           'type': 'string'},
                               'record_as_artifact': {'description': 'Set false for intermediate files that '
                                                                     'should not appear as session artifacts, '
                                                                     'such as HTML used only to print a PDF. '
                                                                     'Defaults to true for artifact-like '
                                                                     'extensions.',
                                                      'type': 'boolean'}},
                'required': ['file_path', 'content'],
                'type': 'object'}}

OPENCODE_PARAMETERS = {'bash': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
          'type': 'object',
          'properties': {'command': {'type': 'string', 'description': 'The command to execute'},
                         'timeout': {'minimum': -9007199254740991,
                                     'exclusiveMinimum': 0,
                                     'type': 'integer',
                                     'maximum': 9007199254740991,
                                     'description': 'Optional timeout in milliseconds'},
                         'workdir': {'type': 'string',
                                     'description': 'The working directory to run the command in. Defaults to '
                                                    "the current directory. Use this instead of 'cd' "
                                                    'commands.'}},
          'required': ['command']},
 'edit': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
          'type': 'object',
          'properties': {'filePath': {'type': 'string',
                                      'description': 'The absolute path to the file to modify'},
                         'oldString': {'type': 'string', 'description': 'The text to replace'},
                         'newString': {'type': 'string',
                                       'description': 'The text to replace it with (must be different from '
                                                      'oldString)'},
                         'replaceAll': {'type': 'boolean',
                                        'description': 'Replace all occurrences of oldString (default false)'}},
          'required': ['filePath', 'oldString', 'newString']},
 'glob': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
          'type': 'object',
          'properties': {'pattern': {'type': 'string',
                                     'description': 'The glob pattern to match files against'},
                         'path': {'type': 'string',
                                  'description': 'The directory to search in. If not specified, the current '
                                                 'working directory will be used. IMPORTANT: Omit this field '
                                                 'to use the default directory. DO NOT enter "undefined" or '
                                                 '"null" - simply omit it for the default behavior. Must be a '
                                                 'valid directory path if provided.'}},
          'required': ['pattern']},
 'grep': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
          'type': 'object',
          'properties': {'pattern': {'type': 'string',
                                     'description': 'The regex pattern to search for in file contents'},
                         'path': {'type': 'string',
                                  'description': 'The directory to search in. Defaults to the current working '
                                                 'directory.'},
                         'include': {'type': 'string',
                                     'description': 'File pattern to include in the search (e.g. "*.js", '
                                                    '"*.{ts,tsx}")'}},
          'required': ['pattern']},
 'read': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
          'type': 'object',
          'properties': {'filePath': {'type': 'string',
                                      'description': 'The absolute path to the file or directory to read'},
                         'offset': {'minimum': 0,
                                    'type': 'integer',
                                    'maximum': 9007199254740991,
                                    'description': 'The line number to start reading from (1-indexed)'},
                         'limit': {'minimum': 0,
                                   'type': 'integer',
                                   'maximum': 9007199254740991,
                                   'description': 'The maximum number of lines to read (defaults to 2000)'}},
          'required': ['filePath']},
 'write': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
           'type': 'object',
           'properties': {'content': {'type': 'string', 'description': 'The content to write to the file'},
                          'filePath': {'type': 'string',
                                       'description': 'The absolute path to the file to write (must be '
                                                      'absolute, not relative)'}},
           'required': ['content', 'filePath']}}
# fmt: on
