import logging

from gui.core import *
from utils import asyncjob


class JobOverlay(GUIElement):

    def gui(self):
        job = asyncjob.current_job()
        if self.state.current_job != job:
            self.state.current_job = job
            if job:
                imgui.open_popup(WindowName.POPUP_JOB.value)
        self.popup_job(job)

    def popup_job(self, job: asyncjob.AsyncJob):
        if imgui.begin_popup_modal(WindowName.POPUP_JOB.value,
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                                         imgui.WINDOW_NO_RESIZE |
                                         imgui.WINDOW_NO_MOVE)[0]:
            if job is None:
                logging.debug("Job no longer exist, closing popup")
                imgui.close_current_popup()
            else:
                imgui.label_text("Job Name", job.params.name or "N/A")
                imgui.separator()
                imgui.label_text("Progress", f"{job.progress.percentage:.0f}%")
            imgui.end_popup()
