"""Simple expyriment text box without leading whitespace stripping."""

import re
from expyriment.stimuli import TextBox

class CustomTextBox(TextBox):
    """
    This class is a copy of the TextBox class from expyriment.stimuli.textbox
    with the only difference that it does not strip leading whitespace from the text.
    This allows simpler text formatting of the script file using a constant text box size.

    This code has been commented out twice:
    # while lines and not lines[0]:
    #     del lines[0]
    """

    def format_block(self, block):
        """Format the given block of text.

        This function is trimming leading and trailing
        empty lines and any leading whitespace that is common to all lines.

        Parameters
        ----------
        block : str
            block of text to be formatted

        """

        # Separate block into lines
        # lines = str(block).split('\n')
        lines = block.split("\n")

        # Remove leading/trailing empty lines
        # while lines and not lines[0]:
        #     del lines[0]
        while lines and not lines[-1]:
            del lines[-1]

        # Look at first line to see how much indentation to trim
        try:
            ws = re.match(r"\s*", lines[0]).group(0)
        except Exception:
            ws = None
        if ws:
            lines = [x.replace(ws, "", 1) for x in lines]

        # Remove leading/trailing blank lines (after leading ws removal)
        # We do this again in case there were pure-whitespace lines
        # while lines and not lines[0]:
        #     del lines[0]
        while lines and not lines[-1]:
            del lines[-1]
        return "\n".join(lines) + "\n"

    # End of code taken from the word-wrapped text display module
