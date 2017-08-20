import sys
import time

class Progressbar(object):

    def __init__(self, message='Progress:', update_frequency_ms=500):
        """
        :param message: short title to be printed before the progress bar
        :param update_frequency_ms: update progressbar at most once every
                                    update_frequency_ms
        """
        self.message = message
        self.update_frequency_ms = update_frequency_ms
        self.last_update_ms = time.time() * 1000

    # update_progress() : Displays or updates a console progress bar
    # Accepts two floats: amount done and the total amount (converted to float).
    #         boolean	: whether to force progress update
    # A value under 0 represents a 'halt'.
    # A value at 1 or bigger represents 100%
    def update_progress(self, done, total, force_update = False):
        curr_time = time.time() * 1000
        if curr_time - self.last_update_ms < self.update_frequency_ms and not force_update:
            return
        self.last_update_ms = curr_time

        barLength = 25  # Modify this to change the length of the progress bar
        status = ""
        progress = float("{0:.3f}".format(done / float(total)))
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\r{3}: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), progress * 100, status, self.message)
        text += ' (%s/%s) ' % (done, total)
        sys.stdout.write(text)
        sys.stdout.flush()
