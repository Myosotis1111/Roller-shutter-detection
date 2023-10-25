import os
import shutil
from math import sqrt
import openpyxl
from model import Model


class Evaluation:

    def __init__(self):

        self.workbook = None
        self.sheet = None
        self.right_count = 0  # correct status predicted
        self.wrong_count = 0  # incorrect status predicted
        self.average_error_total = 0
        self.right_count_sum = 0
        self.wrong_count_sum = 0
        self.average_error = 0  # For "closed" status, the difference between predicted DoC and ground truth
        self.standard_deviation = 0  # the standard deviation between predicted DoC and ground truth
        self.subdir_path = None
        self.folder_path = 'testset'  # path of data set to be evaluated
        self.folder_name = ""  # current subdir
        self.folder_number = 0
        self.fileNames = []  # filenames of images in current subdir
        self.model = Model()
        self.model.mode = "test"  # test mode will exclude some GUI updating in the original method form model.py
        self.model.conf_thres = 0.25
        self.model.iou_thres = 0.45
        self.window_no = None
        # count of scenarios when {predicted status}_{ground truth status}
        # e.g., for window[upper_1], predicted status is open, while the ground truth is closed, then open_closed + 1
        self.open_open = 0
        self.open_closed = 0
        self.open_blocked = 0
        self.closed_open = 0
        self.closed_closed = 0
        self.closed_blocked = 0
        self.blocked_open = 0
        self.blocked_closed = 0
        self.blocked_blocked = 0
        # count of ground truth scenarios when both open and closed status are detected in two camera positions
        self.oc_fusion_closed = 0
        self.oc_fusion_open = 0
        self.oc_fusion_blocked = 0

    def evaluation(self):

        # clear previous results
        result_folder = "evaluation/result"
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        os.makedirs(result_folder)

        tmp_folder = "evaluation/tmp"
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)

        sheet = self.sheet

        # add the table head to the Excel sheet
        head = ["time", "correct prediction", "wrong prediction", "detection accuracy",
                "error of DoC", "standard deviation of error"]

        sheet.append(head)

        for subdir in os.listdir(self.folder_path):
            self.folder_number += 1
            self.subdir_path = os.path.join(self.folder_path, subdir)
            if os.path.isdir(self.subdir_path):
                self.folder_name = subdir
                fileNames = []
                for file in os.listdir(self.subdir_path):
                    file_path = os.path.join(self.subdir_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fileNames.append(file_path)

                # reuse of the method in model.py: initialize the window instance when uploading
                self.model.initializeWindows()

                # reuse of the method in model.py: crop images according to coordinate text file
                self.model.uploadPreprocessing(fileNames, self.model.mode)

                # reuse of the method in model.py: detection process
                self.model.detect(self.model.mode)

                # compare the prediction with the ground truth and get desired metrics
                self.groundTruthComparison()

                line = [self.folder_name, int(self.right_count), int(self.wrong_count),
                        float((self.right_count / (self.wrong_count + self.right_count))),
                        float(self.average_error), float(self.standard_deviation)]

                sheet.append(line)

    def groundTruthComparison(self):

        txt_filename = self.folder_name + ".txt"
        txt_path = os.path.join(self.subdir_path, txt_filename)
        error_sum = 0
        error_count = 0
        error_array = []
        self.right_count = 0
        self.wrong_count = 0
        self.average_error = 0

        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split()
                    window_no = parts[0]
                    status = parts[1]
                    if status == "closed":
                        doc = float(parts[2]) if len(parts) >= 3 else None

                    if not status == "not_detected":
                        if self.model.windows[window_no].status == "open" and status == "open":
                            self.open_open += 1
                            self.right_count += 1
                        if self.model.windows[window_no].status == "open" and status == "closed":
                            self.open_closed += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "open" and status == "blocked":
                            self.open_blocked += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "closed" and status == "open":
                            self.closed_open += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "closed" and status == "closed":
                            self.closed_closed += 1
                            error = abs(self.model.windows[window_no].doc - doc)

                            # if error is over 10, then we consider it failure of detection
                            if error <= 10:
                                self.right_count += 1
                                error_count += 1
                                error_sum += error
                                error_array.append(error)
                            else:
                                self.wrong_count += 1

                        if self.model.windows[window_no].status == "closed" and status == "blocked":
                            self.closed_blocked += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "blocked" and status == "open":
                            self.blocked_open += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "blocked" and status == "closed":
                            self.blocked_closed += 1
                            self.wrong_count += 1
                        if self.model.windows[window_no].status == "blocked" and status == "blocked":
                            self.blocked_blocked += 1
                            self.right_count += 1
                        if self.model.windows[window_no].count_open == 1 and \
                           self.model.windows[window_no].count_closed == 1:
                            if status == "closed":
                                self.oc_fusion_closed += 1
                            if status == "open":
                                self.oc_fusion_open += 1
                            if status == "blocked":
                                self.oc_fusion_blocked += 1

        self.average_error = error_sum / error_count
        self.standard_deviation = sqrt(sum((x - self.average_error) ** 2 for x in error_array) / len(error_array))
        self.average_error_total += self.average_error
        self.right_count_sum += self.right_count
        self.wrong_count_sum += self.wrong_count


evaluation = Evaluation()
# create a new Excel workbook
evaluation.workbook = openpyxl.Workbook()

evaluation.sheet = evaluation.workbook.active

evaluation.evaluation()

evaluation.sheet.append([])

# adding overall results
result_line = ["Result:", int(evaluation.right_count_sum), int(evaluation.wrong_count_sum),
               float(evaluation.right_count_sum / (evaluation.wrong_count_sum + evaluation.right_count_sum)),
               float(evaluation.average_error_total / evaluation.folder_number)]

evaluation.sheet.append(result_line)

# adding confusion matrix
data = [
    ["pred/gt", 'open', 'closed', 'blocked'],
    ['open', evaluation.open_open, evaluation.open_closed, evaluation.open_blocked],
    ['closed', evaluation.closed_open, evaluation.closed_closed, evaluation.closed_blocked],
    ['blocked', evaluation.blocked_open, evaluation.blocked_closed, evaluation.blocked_blocked],
    ['open_close_fusion', evaluation.oc_fusion_open, evaluation.oc_fusion_closed, evaluation.oc_fusion_blocked]
]

evaluation.sheet.append([])

for row in data:
    evaluation.sheet.append(row)

# save as Excel file
evaluation.workbook.save('evaluation/result.xlsx')
