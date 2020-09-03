# https://github.com/aegirhall/console-menu
# Import the necessary packages
import os
from consolemenu import *
from consolemenu.items import *
from load_data import


pipeline_steps = [
    "Data Cleaning",
    "Feature Engineering",
    "Custom Feature Engineering",
    "Modeling"
]


def remove_safe(lst, element):
    """
    Delete a list element, if it does not exist don't throw an error

    :param lst: list
    :param element: element to be removed
    :return: list
    """

    try:
        lst.remove(element)
    except ValueError:
        pass

    return lst


def list_projects():
    """
    Returns list of all project to select

    :return: list of projects
    """

    return remove_safe(os.listdir("projects/"), ".DS_Store")


def get_project_path(project):
    """
    Change the current_directory to be for the project selected by the user

    :param project: str
    """
    return menu.selected_option


# Create the menu
menu = ConsoleMenu("Kaggle Utility", prologue_text="Welcome to the Kaggle Utility, a user-friendly tool to quickly "
                                                   "implement a data pipeline. Select a project to begin:")

# A SelectionMenu constructs a menu from a list of strings
pipeline_menu = SelectionMenu(pipeline_steps, prologue_text="Select the part of the pipeline you want to work on for "
                                                            "this project:")

for proj in list_projects():
    # A SubmenuItem lets you add a menu (the selection_menu above, for example)
    # as a submenu of another menu
    menu.append_item(SubmenuItem(f"{proj}", pipeline_menu, menu))

cleaning_menu = ConsoleMenu("Data Cleaning", prologue_text="Select from potential options:")
print_logs_item = CommandItem("Print logs", "echo Logs printed")
load_data_item = FunctionItem("Call a Python function", input, ["Enter an input"])


# Finally, we call show to show the menu and allow the user to interact
menu.show()
