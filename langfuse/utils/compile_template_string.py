def compile_template_string(content: str, **kwargs) -> str:
    opening = "{{"
    closing = "}}"

    result_list = []
    curr_idx = 0

    while curr_idx < len(content):
        # Find the next opening tag
        var_start = content.find(opening, curr_idx)

        if var_start == -1:
            result_list.append(content[curr_idx:])
            break

        # Find the next closing tag
        var_end = content.find(closing, var_start)

        if var_end == -1:
            result_list.append(content[curr_idx:])
            break

        # Append the content before the variable
        result_list.append(content[curr_idx:var_start])

        # Extract the variable name
        variable_name = content[var_start + len(opening) : var_end].strip()

        # Append the variable value
        if variable_name in kwargs:
            result_list.append(str(kwargs[variable_name]))
        else:
            result_list.append(content[var_start : var_end + len(closing)])

        curr_idx = var_end + len(closing)

    return "".join(result_list)
