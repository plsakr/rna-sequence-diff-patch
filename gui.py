import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication, QPushButton, QLineEdit, QTableWidget, QComboBox, QTableWidgetItem,
                               QTabWidget, QLabel, QListWidget, QAbstractItemView, QFileDialog, QRadioButton)

from PySide6.QtCore import QFile, QIODevice, Qt
from PySide6.QtGui import QColor
from StringEditDistance import wagnerFisher, create_paths, generate_es, patching, generate_rev_es, reload_user_costs, user_costs
import json

import fa_import


ui_file_name = "mainwindow.ui"
ui_table_name = "costtable.ui"
ui_dataset_name = 'dataset.ui'

nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']

sequence1 = ''
sequence2 = ''
dp = [[]]
paths = []
current_path = []
es = []

inverted = False
prog_inv = False
force_refresh_table = False

edit_scripts = []


def validate_sequence(seq):
    global nucleotides
    for ch in seq:
        if ch in nucleotides:
            continue
        else:
            return False
    return True


def format_edit_script(es):
    out = '['
    for op in es:
        operation = op[0]

        if operation == 'insert':
            out += f'Ins({op[1]},{op[2][1]}),'
        elif operation == 'delete':
            out += f'Del({op[1][0]}),'
        else:
            out += f'Upd({op[1][0]},{op[2][1]}),'

    if len(out) > 1:
        out = out[:-1]
    out += ']'
    return out


def import_from_dataset(sequenceNbr: int, window, list_widget: QListWidget, btn: QPushButton, seq1: QLineEdit, seq2: QLineEdit):

    def on_import():

        if len(list_widget.selectedItems()) != 0:
            key = list_widget.selectedItems()[0].text()
            seq = fa_import.get_seq(key)
            if sequenceNbr == 1:
                seq1.setText(seq)
            else:
                seq2.setText(seq)
            btn.clicked.disconnect()
            window.hide()

    def on_click():
        window.show()
        btn.clicked.connect(on_import)

    return on_click



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

def on_insert_cost_change(op):

    def on_change(val):
        try:
            score = float(val)
            copy_costs = user_costs
            copy_costs[op] = score

            with open('user_costs.json', 'w') as f:
                json.dump(copy_costs, f)

            reload_user_costs()
        except:
            pass

    return on_change


def on_combo_changed(table):
    global nucleotides

    def on_change(ind):
        print('ana hon')
        if ind > 0:
            val = nucleotides[ind - 1]
            scores = user_costs['update'][val]

            scores = list(scores.values())
            for i in range(len(nucleotides)):
                it = QTableWidgetItem(str(scores[i]))
                table.setItem(i, 1, it)
        else:
            for i in range(len(nucleotides)):
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


def on_es_list_changed(table: QTableWidget, btn_to_patch: QPushButton, btn_export: QPushButton):
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

            # es_label.setText(str(es))

        if ind < 0:
            btn_to_patch.setEnabled(False)
            btn_export.setEnabled(False)

    return on_change


def onTabChanged(rButton: QRadioButton, table: QTableWidget, l_cost: QLabel, l_sim: QLabel, es_list: QListWidget, label_t: QLabel,
                 label_chosen_es: QLabel, radio_1: QRadioButton, radio_2: QRadioButton, button_start_patch: QPushButton, lbl_comp: QLabel,
                 lbl_comp_title: QLabel, list_choose_es: QListWidget):
    global sequence1, sequence2, dp, paths, es, edit_scripts, force_refresh_table

    def on_change(ind):
        global sequence1, sequence2, dp, paths, es, edit_scripts, force_refresh_table
        print('CURRENT TAB', ind)

        if ind == 1:
            if force_refresh_table:
                force_refresh_table = False
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

                if is_user_cost:
                    dp_def = wagnerFisher(sequence1, sequence2, False)
                    def_cost = dp_def[len(sequence1)][len(sequence2)].value
                    label_comparison.setText(str(1 / (1+def_cost)))
                else:
                    label_comparison.hide()
                    label_comparison_title.hide()

                for i in range(len(sequence1) + 1):
                    for j in range(len(sequence2) + 1):
                        cell = QTableWidgetItem(str(round(dp[i][j].value, 2)))
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
                edit_scripts.clear()
                for p in paths:
                    my_es = generate_es(p, sequence1, sequence2)
                    edit_scripts.append(my_es)
                    if len(my_es) == 0:
                        view = f'Edit Script #{i + 1}: {format_edit_script(my_es)} - Edit script is empty --> sequences are already homomorphic'
                    else:
                        view = f'Edit Script #{i + 1}: {format_edit_script(my_es)}'
                    es_list.addItem(view)
                    i += 1

        elif ind == 2:  # we are patching
            if es != []:  # already chosen editscript
                label_chosen_es.setText(f'Chosen Edit Script: {format_edit_script(es)}')
                radio_1.setText(sequence1)
                radio_2.setText(sequence2)
                button_start_patch.setEnabled(True)
            if edit_scripts != []:
                radio_1.setText(sequence1)
                radio_2.setText(sequence2)
                i = 0
                list_choose_es.clear()
                for e in edit_scripts:
                    if len(e) == 0:
                        view = f'Edit Script #{i + 1}: {format_edit_script(e)} - Edit script is empty --> sequences are already homomorphic'
                    else:
                        view = f'Edit Script #{i + 1}: {format_edit_script(e)}'
                    list_choose_es.addItem(view)
                    i += 1

    return on_change


def onInputNextClicked(eText1, eText2, tabs: QTabWidget):
    global sequence1, sequence2, force_refresh_table

    def on_click():
        global sequence1, sequence2, force_refresh_table
        sequence1 = eText1.text()
        sequence2 = eText2.text()
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
            force_refresh_table = True
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


def on_import_patching(list_choose_es: QListWidget, radio_1: QRadioButton, radio_2: QRadioButton):
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
                    view = f'Edit Script #{i + 1}: {format_edit_script(e)} - Edit script is empty --> sequences are already homomorphic'
                else:
                    view = f'Edit Script #{i + 1}: {format_edit_script(e)}'
                list_choose_es.addItem(view)
                i += 1

            radio_1.setText(sequence1)
            radio_2.setText(sequence2)

            print('loaded JSON file')

    return on_click


def on_select_es(es_label: QLabel, start_patch: QPushButton, radio_1: QRadioButton):
    global sequence1, sequence2, es, edit_scripts, inverted, prog_inv

    def on_change(ind):
        global sequence1, sequence2, es, edit_scripts, inverted, prog_inv
        print('ana ma2boor hon v2')

        es = edit_scripts[ind]
        es_label.setText(str(format_edit_script(es)))
        start_patch.setEnabled(True)

        if ind < 0:
            es = []
            es_label.setText('')
            start_patch.setEnabled(False)

        if inverted:
            tmp = sequence1
            sequence1 = sequence2
            sequence2 = tmp
            inverted = not inverted
            prog_inv = True
            radio_1.setChecked(True)


    return on_change


def on_reverse(radio_1: QRadioButton, radio_2: QRadioButton, label_new_es: QLabel):
    global es, sequence1, sequence2, inverted, prog_inv

    def on_click():
        global es, sequence1, sequence2, inverted, prog_inv
        if prog_inv:
            prog_inv = False
            return
        temp = sequence1
        sequence1 = sequence2
        sequence2 = temp
        es = generate_rev_es(es, sequence1, sequence2)
        label_new_es.setText(str(format_edit_script(es)))
        inverted = not inverted

    return on_click


def on_start_patching(radio_1: QRadioButton, radio_2: QRadioButton, label_out: QLabel):
    global es

    def on_click():
        global es
        val = radio_1.text() if radio_1.isChecked() else radio_2.text()

        if len(val) > 0 and es != []:
            print('PATCHING')
            patched = patching(es, val)
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
    if not ui_table_fil.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)

    load_table = QUiLoader()
    table_window = loader.load(ui_table_fil)
    ui_table_fil.close()

    ui_dataset_file = QFile(ui_dataset_name)
    if not ui_dataset_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)

    load_dataset_window = QUiLoader()
    dataset_window = loader.load(ui_dataset_file)
    ui_dataset_file.close()

    if not window:
        print(loader.errorString())
        sys.exit(-1)
    window.show()

    tab_widget = window.findChild(QTabWidget, 'tabWidget')
    tab_widget.setTabEnabled(1, False)
    # main window variables:
    # tab 1:
    next_button = window.findChild(QPushButton, 'button_input_next')
    btn_import_1 = window.findChild(QPushButton, 'btn_import_1')
    btn_import_2 = window.findChild(QPushButton, 'btn_import_2')
    edit_seq1 = window.findChild(QLineEdit, 'edit_sequence1')
    edit_seq2 = window.findChild(QLineEdit, 'edit_sequence2')
    edit_cost_ins = window.findChild(QLineEdit, 'edit_cost_insert')
    edit_cost_del = window.findChild(QLineEdit, 'edit_cost_delete')
    btn_table = window.findChild(QPushButton, 'btn_open_table')
    radio_cost_user = window.findChild(QRadioButton, 'radio_cost_user')
    # cost table:
    cost_table = table_window.findChild(QTableWidget, 'cost_table')
    combo_from = table_window.findChild(QComboBox, 'combo_from')

    # dataset window
    list_dataset = dataset_window.findChild(QListWidget, 'list_data')
    btn_import_data = dataset_window.findChild(QPushButton, 'btn_import_data')

    # tab 2:
    ed_matrix_table = window.findChild(QTableWidget, 'ed_matrix_table')
    label_cost = window.findChild(QLabel, 'label_ec')
    label_sim = window.findChild(QLabel, 'label_sim')
    es_list = window.findChild(QListWidget, 'es_list')
    label_title = window.findChild(QLabel, 'label_calculator')
    btn_to_patching = window.findChild(QPushButton, 'btn_next')
    btn_export_patching = window.findChild(QPushButton, 'btn_export')
    label_comparison_title = window.findChild(QLabel, 'label_comparison_title')
    label_comparison = window.findChild(QLabel, 'label_comparison')

    # tab 3:
    btn_import = window.findChild(QPushButton, 'btn_import')
    list_patch_choose = window.findChild(QListWidget, 'list_patch_es')
    label_es_chosen = window.findChild(QLabel, 'label_es_chosen')
    # edit_patch_input = window.findChild(QLineEdit, 'edit_patch_input')
    radio_patch_seq1 = window.findChild(QRadioButton, 'radio_patch_sequence1')
    radio_patch_seq2 = window.findChild(QRadioButton, 'radio_patch_sequence2')
    btn_patch = window.findChild(QPushButton, 'btn_patch')
    label_patched = window.findChild(QLabel, 'label_patched')
    # btn_rev = window.findChild(QPushButton, 'btn_invert')

    edit_cost_ins.setText(str(user_costs['insert']))
    edit_cost_del.setText(str(user_costs['delete']))

    combo_selections = ['Please Select Nucleotide...', *nucleotides]
    combo_from.insertItems(0, combo_selections)

    index = 0
    for n in nucleotides:
        item = QTableWidgetItem(n)
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        cost_table.setItem(index, 0, item)
        index += 1

    all_titles = fa_import.get_all_keys()
    for t in all_titles:
        list_dataset.addItem(t)

    # add all listeners
    # tab 1:
    btn_table.clicked.connect(lambda: table_window.show())
    btn_import_1.clicked.connect(import_from_dataset(1, dataset_window, list_dataset, btn_import_data, edit_seq1, edit_seq2))
    btn_import_2.clicked.connect(import_from_dataset(2, dataset_window, list_dataset, btn_import_data, edit_seq1, edit_seq2))

    edit_seq1.textEdited.connect(onSequence1Changed(edit_seq1))
    edit_seq2.textEdited.connect(onSequence2Changed(edit_seq2))
    next_button.clicked.connect(onInputNextClicked(edit_seq1, edit_seq2, tab_widget))
    tab_widget.currentChanged.connect(
        onTabChanged(radio_cost_user, ed_matrix_table, label_cost, label_sim, es_list, label_title,
                     label_es_chosen, radio_patch_seq1, radio_patch_seq2,
                     btn_patch, label_comparison, label_comparison_title, list_patch_choose))
    edit_cost_ins.textEdited.connect(on_insert_cost_change('insert'))
    edit_cost_del.textEdited.connect(on_insert_cost_change('delete'))
    combo_from.currentIndexChanged.connect(on_combo_changed(cost_table))
    cost_table.cellChanged.connect(on_table_edited(combo_from, cost_table))

    # tab 2:
    es_list.currentRowChanged.connect(
        on_es_list_changed(ed_matrix_table, btn_to_patching, btn_export_patching))
    btn_to_patching.clicked.connect(on_goto_patching(tab_widget))

    # tab 3:
    btn_patch.clicked.connect(on_start_patching(radio_patch_seq1, radio_patch_seq2, label_patched))
    btn_export_patching.clicked.connect(on_export_to_file)
    btn_import.clicked.connect(on_import_patching(list_patch_choose, radio_patch_seq1, radio_patch_seq2))
    list_patch_choose.currentRowChanged.connect(on_select_es(label_es_chosen, btn_patch, radio_patch_seq1))

    radio_patch_seq1.toggled.connect(on_reverse(radio_patch_seq1, radio_patch_seq2, label_es_chosen))

    sys.exit(app.exec_())
