import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication, QPushButton, QLineEdit, QTableWidget, QComboBox, QTableWidgetItem,
                               QTabWidget, QLabel, QListWidget, QAbstractItemView, QFileDialog, QRadioButton)

from PySide6.QtCore import QFile, QIODevice, Qt
from PySide6.QtGui import QColor
from StringEditDistance import wagnerFisher, create_paths, generate_es, patching, generate_rev_es, reload_user_costs, user_costs
import json

ui_file_name = "mainwindow.ui"
ui_table_name = "costtable.ui"

nucleotides = ['A', 'G', 'C', 'U', 'R', 'M', 'S', 'V', 'N']

sequence1 = ''
sequence2 = ''
dp = [[]]
paths = []
current_path = []
es = []

inverted = False

edit_scripts = []


def validate_sequence(seq):
    global nucleotides
    for ch in seq:
        if ch in nucleotides:
            continue
        else:
            return False
    return True


def onSequence1Changed(eText):
    global sequence1

    def on_change(val):
        global sequence1
        sequence1 = val
        if len(sequence1) == 0 or not(validate_sequence(val)):
            eText.setStyleSheet('QLineEdit{ border-width: 1px; border-style: solid; border-color:  red;}')
        else:
            eText.setStyleSheet('')

    return on_change


def onSequence2Changed(eText):
    global sequence2

    def on_change(val):
        global sequence2
        sequence2 = val
        if len(sequence2) == 0 or not(validate_sequence(val)):
            eText.setStyleSheet('QLineEdit{ border-width: 1px; border-style: solid; border-color:  red;}')
        else:
            eText.setStyleSheet('')

    return on_change


def on_combo_changed(table):
    global nucleotides

    def on_change(ind):
        print('ana hon')
        if ind > 0:
            val = nucleotides[ind - 1]
            scores = user_costs['update'][val]

            scores = list(scores.values())
            for i in range(9):
                it = QTableWidgetItem(str(scores[i]))
                table.setItem(i, 1, it)
        else:
            for i in range(9):
                it = QTableWidgetItem()
                table.setItem(i, 1, it)

    return on_change


def on_table_edited(combo, table):
    global nucleotides

    def on_change(row, col):
        global nucleotides
        print(f'edited ({row},{col})')
        currentNucleotideFrom = combo.currentIndex() - 1

        if currentNucleotideFrom >= 0:
            print(user_costs)
            currentNuc = nucleotides[currentNucleotideFrom]
            copy_costs = user_costs
            copy_costs['update'][currentNuc][nucleotides[row]] = float(table.item(row, col).text())
            with open('user_costs.json', 'w') as f:
                json.dump(copy_costs, f)

            reload_user_costs()
            print(user_costs)
            print('DEFAULT COSTS UPDATED. ')

    return on_change


def on_es_list_changed(es_label: QLabel, table: QTableWidget, btn_to_patch: QPushButton, btn_export: QPushButton):
    global current_path, paths, sequence1, sequence2, es

    def on_change(ind):
        global current_path, paths, sequence1, sequence2, es
        print('ana ma2boor hon')

        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                table.item(i, j).setBackground(QColor(255, 255, 255))
                print(f'{i}, {j}')

        if ind >= 0:
            p = paths[ind]

            for n in p:
                i = n.i + 1
                j = n.j + 1
                table.item(i, j).setBackground(QColor(174, 224, 123))
                print(f'{i}, {j}')

            current_path = p
            es = generate_es(p, sequence1, sequence2)

            if len(es) != 0:
                btn_to_patch.setEnabled(True)
                btn_export.setEnabled(True)

            es_label.setText(str(es))

        if ind < 0:
            btn_to_patch.setEnabled(False)
            btn_export.setEnabled(False)

    return on_change


def onTabChanged(rButton: QRadioButton, table: QTableWidget, l_cost: QLabel, l_sim: QLabel, es_list: QListWidget, label_t: QLabel,
                 label_chosen_es: QLabel, line_patch_input: QLineEdit, button_start_patch: QPushButton):
    global sequence1, sequence2, dp, paths, es

    def on_change(ind):
        global sequence1, sequence2, dp, paths, es
        print('CURRENT TAB', ind)

        if ind == 1:
            label_t.setText(f'Cost table of transforming "{sequence1}" into "{sequence2}":')
            table.clear()
            table.setRowCount(len(sequence1) + 1)
            table.setColumnCount(len(sequence2) + 1)
            table.setVerticalHeaderLabels(['', *sequence1])
            table.setHorizontalHeaderLabels(['', *sequence2])
            table.horizontalHeader().setStyleSheet('* {font-weight: bold;}')
            table.verticalHeader().setStyleSheet('* {font-weight: bold;}')

            # DO WAGNER FISHER
            is_user_cost = rButton.isChecked()
            dp = wagnerFisher(sequence1, sequence2, is_user_cost)

            for i in range(len(sequence1) + 1):
                for j in range(len(sequence2) + 1):
                    cell = QTableWidgetItem(str(dp[i][j].value))
                    cell.setFlags(cell.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                    cell.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, j, cell)

            e_cost = dp[len(sequence1)][len(sequence2)].value
            l_cost.setText(str(e_cost))
            l_sim.setText(str(1 / (1 + e_cost)))
            es_list.setSelectionMode(QAbstractItemView.SingleSelection)

            # Generate edit scripts
            paths = create_paths(dp)
            i = 0
            es_list.clear()
            for p in paths:
                my_es = generate_es(p, sequence1, sequence2)
                if len(my_es) == 0:
                    view = f'Edit Script #{i + 1}: {my_es} - Edit script is empty --> sequences are already homomorphic'
                else:
                    view = f'Edit Script #{i + 1}: {my_es}'
                es_list.addItem(view)
                i += 1

        elif ind == 2:  # we are patching
            if es != []:  # already chosen editscript
                label_chosen_es.setText(f'Chosen Edit Script: {es}')
                line_patch_input.setText(sequence1)
                button_start_patch.setEnabled(True)

    return on_change


def onInputNextClicked(eText1, eText2, tabs: QTabWidget):
    global sequence1, sequence2

    def on_click():
        global sequence1, sequence2
        if len(sequence1) == 0 or len(sequence2) == 0 or not(validate_sequence(sequence1)) or not(validate_sequence(sequence2)):
            print('ERROR FOUND: SEQUENCES NOT VALID')
            if len(sequence1) == 0 or not(validate_sequence(sequence1)):
                eText1.setStyleSheet('QLineEdit{ border-width: 1px; border-style: solid; border-color:  red;}')
            else:
                eText1.setStyleSheet('')
            if len(sequence2) == 0 or not(validate_sequence(sequence2)):
                eText2.setStyleSheet('QLineEdit{ border-width: 1px; border-style: solid; border-color:  red;}')
            else:
                eText2.setStyleSheet('')
        else:
            print('Inputs and costs confirmed:')
            print(f'Seq1: {sequence1}. Seq2: {sequence2}')
            tabs.setTabEnabled(1, True)
            tabs.setCurrentIndex(1)

    return on_click


def on_export_to_file():
    global sequence1, sequence2, paths

    save_path_and_ext = QFileDialog.getSaveFileName(filter='JSON (*.json)')
    save_path = save_path_and_ext[0]
    if save_path != '':
        # print(save_path)
        obj_to_save = {'seq1': sequence1, 'seq2': sequence2, 'es': []}

        for p in paths:
            my_es = generate_es(p, sequence1, sequence2)
            if len(my_es) > 0:
                obj_to_save['es'].append(my_es)

        with open(save_path, 'w') as f:
            json.dump(obj_to_save, f, indent=4)


def on_goto_patching(tabs: QTabWidget):
    def on_click():
        tabs.setCurrentIndex(2)

    return on_click


def on_import_patching(list_choose_es: QListWidget, line_input: QLineEdit):
    global sequence1, sequence2, edit_scripts

    def on_click():
        global sequence1, sequence2, edit_scripts
        open_path_ext = QFileDialog.getOpenFileName(filter='JSON (*.json)')
        open_path = open_path_ext[0]

        if open_path != '':
            with open(open_path, 'r') as f:
                data = json.load(f)
                sequence1 = data['seq1']
                sequence2 = data['seq2']
                edit_scripts = data['es']

            i = 0
            list_choose_es.clear()
            for e in edit_scripts:
                if len(e) == 0:
                    view = f'Edit Script #{i + 1}: {e} - Edit script is empty --> sequences are already homomorphic'
                else:
                    view = f'Edit Script #{i + 1}: {e}'
                list_choose_es.addItem(view)
                i += 1

            line_input.setText(sequence1)

            print('loaded JSON file')

    return on_click


def on_select_es(es_label: QLabel, start_patch: QPushButton, seq_inpu: QLineEdit):
    global sequence1, sequence2, es, edit_scripts, inverted

    def on_change(ind):
        global sequence1, sequence2, es, edit_scripts, inverted
        print('ana ma2boor hon v2')

        es = edit_scripts[ind]
        es_label.setText(str(es))
        start_patch.setEnabled(True)

        if ind < 0:
            es = []
            es_label.setText('')
            start_patch.setEnabled(False)

        if inverted:
            tmp = sequence1
            sequence1 = sequence2
            sequence2 = tmp
            seq_inpu.setText(sequence1)
            inverted = not inverted

    return on_change


def on_reverse(line_patch_input: QLineEdit, label_new_es: QLabel):
    global es, sequence1, sequence2, inverted

    def on_click():
        global es, sequence1, sequence2, inverted
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        es = generate_rev_es(es, sequence1, sequence2)
        label_new_es.setText(str(es))
        line_patch_input.setText(sequence1)
        inverted = not inverted

    return on_click


def on_start_patching(line_patch_input: QLineEdit, label_out: QLabel):
    global es, sequence2

    def on_click():
        global es, sequence2
        val = line_patch_input.text()

        if len(val) > 0 and es != [] and sequence2 != '':
            print('PATCHING')
            patched = patching(es, val, sequence2)
            label_out.setText(patched)

    return on_click


if __name__ == "__main__":
    # initial app loading
    app = QApplication(sys.argv)

    # load main window
    ui_file = QFile(ui_file_name)
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)
    loader = QUiLoader()
    window = loader.load(ui_file)
    ui_file.close()

    # load table window
    ui_table_fil = QFile(ui_table_name)
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)

    load_table = QUiLoader()
    table_window = loader.load(ui_table_fil)
    ui_table_fil.close()

    if not window:
        print(loader.errorString())
        sys.exit(-1)
    window.show()

    tab_widget = window.findChild(QTabWidget, 'tabWidget')
    tab_widget.setTabEnabled(1, False)
    # main window variables:
    # tab 1:
    next_button = window.findChild(QPushButton, 'button_input_next')
    edit_seq1 = window.findChild(QLineEdit, 'edit_sequence1')
    edit_seq2 = window.findChild(QLineEdit, 'edit_sequence2')
    edit_cost_ins = window.findChild(QLineEdit, 'edit_cost_insert')
    edit_cost_del = window.findChild(QLineEdit, 'edit_cost_delete')
    btn_table = window.findChild(QPushButton, 'btn_open_table')
    radio_cost_user = window.findChild(QRadioButton, 'radio_cost_user')
    # cost table:
    cost_table = table_window.findChild(QTableWidget, 'cost_table')
    combo_from = table_window.findChild(QComboBox, 'combo_from')

    # tab 2:
    ed_matrix_table = window.findChild(QTableWidget, 'ed_matrix_table')
    label_cost = window.findChild(QLabel, 'label_ec')
    label_sim = window.findChild(QLabel, 'label_sim')
    es_list = window.findChild(QListWidget, 'es_list')
    label_es = window.findChild(QLabel, 'label_es')
    label_title = window.findChild(QLabel, 'label_calculator')
    btn_to_patching = window.findChild(QPushButton, 'btn_next')
    btn_export_patching = window.findChild(QPushButton, 'btn_export')

    # tab 3:
    btn_import = window.findChild(QPushButton, 'btn_import')
    list_patch_choose = window.findChild(QListWidget, 'list_patch_es')
    label_es_chosen = window.findChild(QLabel, 'label_es_chosen')
    edit_patch_input = window.findChild(QLineEdit, 'edit_patch_input')
    btn_patch = window.findChild(QPushButton, 'btn_patch')
    label_patched = window.findChild(QLabel, 'label_patched')
    btn_rev = window.findChild(QPushButton, 'btn_invert')

    combo_selections = ['Please Select Nucleotide...', *nucleotides]
    combo_from.insertItems(0, combo_selections)

    index = 0
    for n in nucleotides:
        item = QTableWidgetItem(n)
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        cost_table.setItem(index, 0, item)
        index += 1

    # add all listeners
    # tab 1:
    btn_table.clicked.connect(lambda: table_window.show())

    edit_seq1.textEdited.connect(onSequence1Changed(edit_seq1))
    edit_seq2.textEdited.connect(onSequence2Changed(edit_seq2))
    next_button.clicked.connect(onInputNextClicked(edit_seq1, edit_seq2, tab_widget))
    tab_widget.currentChanged.connect(
        onTabChanged(radio_cost_user, ed_matrix_table, label_cost, label_sim, es_list, label_title, label_es_chosen, edit_patch_input, btn_patch))

    combo_from.currentIndexChanged.connect(on_combo_changed(cost_table))
    cost_table.cellChanged.connect(on_table_edited(combo_from, cost_table))

    # tab 2:
    es_list.currentRowChanged.connect(
        on_es_list_changed(label_es, ed_matrix_table, btn_to_patching, btn_export_patching))
    btn_to_patching.clicked.connect(on_goto_patching(tab_widget))

    # tab 3:
    btn_patch.clicked.connect(on_start_patching(edit_patch_input, label_patched))
    btn_export_patching.clicked.connect(on_export_to_file)
    btn_import.clicked.connect(on_import_patching(list_patch_choose, edit_patch_input))
    list_patch_choose.currentRowChanged.connect(on_select_es(label_es_chosen, btn_patch, edit_patch_input))
    btn_rev.clicked.connect(on_reverse(edit_patch_input, label_es_chosen))

    sys.exit(app.exec_())