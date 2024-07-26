from bertviz import head_view, model_view

def save_head_view(attention, tokens,name_idx=""):


    html_page = head_view(attention, tokens, html_action="return")

    with open("head_view.html", "w") as file:
        file.write(html_page.data)


    html_page = model_view(attention, tokens, html_action="return")

    with open(name_idx+"model_view.html", "w") as file:
        file.write(html_page.data)

def save_model_view(attention, tokens,name_idx=""):
    html_page = model_view(attention, tokens, html_action="return")


    with open(name_idx+"_view.html", "w") as file:
        file.write(html_page.data)
