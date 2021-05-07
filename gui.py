import math
import sys
import time
from operator import itemgetter

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import (QApplication, QPushButton, QLineEdit, QTableWidget, QComboBox, QTableWidgetItem,
                               QTabWidget, QLabel, QListWidget, QAbstractItemView, QFileDialog, QRadioButton,
                               QMessageBox, QCheckBox)

from PySide2.QtCore import QFile, QIODevice, Qt
from PySide2.QtGui import QColor
from StringEditDistance import wagnerFisher, create_paths, generate_es, patching, generate_rev_es, reload_user_costs, user_costs
import json
from widgets import CheckableComboBox
from widgets.mplwidget import MplWidget

from IRMethods import (cosine, pearson, euclidian_distance, manhattan_distance, tanimoto_distance, dice_dist,
                       set_intersection_similarity, set_dice_similarity, set_jaccard_similarity,
                       multi_intersection_similarity, multi_dice_similarity, multi_jaccard_similarity, convert_to_set,
                       convert_to_tf_vector, convert_to_idf_vector, create_tf_idf_vector,
                       convert_to_multi_set, create_and_start_threads, wf_score, create_search_threads, search_collection)

import fa_import
from import_xml import import_xml
from pymongo import MongoClient


ui_file_name = "mainwindow.ui"
ui_table_name = "costtable.ui"
ui_dataset_name = 'dataset.ui'
ui_xml_name = 'xml_import.ui'

nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']

vector_methods_names = [cosine, pearson, manhattan_distance, euclidian_distance, tanimoto_distance, dice_dist]
vector_selections = ['Cosine', 'Pearson', 'Manattan Distance', 'Euclidean Distance', 'Tanimoto', 'Dice']
set_methods_names = [set_intersection_similarity, set_jaccard_similarity, set_dice_similarity]
set_selections = ['Intersection', 'Jaccard', 'Dice']
multiset_methods_names = [multi_intersection_similarity, multi_jaccard_similarity, multi_dice_similarity]
multiset_selections = ['Intersection', 'Jaccard', 'Dice']

sequence1 = ''
sequence2 = ''
dp = [[]]
paths = []
current_path = []
es = []

imported_sequences = {}

inverted = False
prog_inv = False
force_refresh_table = False

edit_scripts = []

similarity_results = {}

collection = MongoClient().rna_db.sequences

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
        operation = op['operation']

        if operation == 'update' and op['source']['character'] == op['destination']['character']:
            continue

        if operation == 'insert':
            out += f'Ins({op["source"]["index"]},{op["destination"]["character"]}),'
        elif operation == 'delete':
            out += f'Del({op["source"]["index"]}),'
        else:
            out += f'Upd({op["source"]["index"]},{op["destination"]["character"]}),'

    if len(out) > 1:
        out = out[:-1]
    out += ']'
    return out

def showdialog(text, is_error = False):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical if is_error else QMessageBox.Warning)

    msg.setText(text)
    # msg.setInformativeText("This is additional information")
    msg.setWindowTitle("Patching Information")
    msg.setStandardButtons(QMessageBox.Ok)

    msg.exec_()


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


def on_browse_xml(list_widget: QListWidget):
    global imported_sequences

    def on_browse():
        global imported_sequences
        open_path_ext = QFileDialog.getOpenFileName(filter='XML (*.xml)')
        open_path = open_path_ext[0]
        if open_path != '':
            imported_sequences = import_xml(open_path)
            list_widget.clear()

            for t in imported_sequences.keys():
                list_widget.addItem(t)

    return on_browse

def import_from_xml(sequenceNbr: int, window, list_widget: QListWidget, btn_save: QPushButton,
                    seq1: QLineEdit, seq2: QLineEdit):
    global imported_sequences


    def on_import():
        global imported_sequences
        if len(list_widget.selectedItems()) != 0:
            key = list_widget.selectedItems()[0].text()
            seq = imported_sequences[key]
            if sequenceNbr == 1:
                seq1.setText(seq)
            else:
                seq2.setText(seq)
            btn_save.clicked.disconnect()
            window.hide()

    def on_click():
        window.show()
        btn_save.clicked.connect(on_import)

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
        # print('ana hon')
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
        # print(f'edited ({row},{col})')
        currentNucleotideFrom = combo.currentIndex() - 1

        if currentNucleotideFrom >= 0:
            # print(user_costs)
            currentNuc = nucleotides[currentNucleotideFrom]
            copy_costs = user_costs
            copy_costs['update'][currentNuc][nucleotides[row]] = float(table.item(row, col).text())
            with open('user_costs.json', 'w') as f:
                json.dump(copy_costs, f)

            reload_user_costs()
            # print(user_costs)
            # print('DEFAULT COSTS UPDATED. ')

    return on_change


def on_es_list_changed(table: QTableWidget, btn_to_patch: QPushButton, btn_export: QPushButton):
    global current_path, paths, sequence1, sequence2, es

    def on_change(ind):
        global current_path, paths, sequence1, sequence2, es

        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                table.item(i, j).setBackground(QColor(255, 255, 255))
                # print(f'{i}, {j}')

        if ind >= 0:
            p = paths[ind]

            for n in p:
                i = n.i + 1
                j = n.j + 1
                table.item(i, j).setBackground(QColor(174, 224, 123))
                # print(f'{i}, {j}')

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
                 label_chosen_es: QLabel, line_edit_patch: QLineEdit, button_start_patch: QPushButton, lbl_comp: QLabel,
                 lbl_comp_title: QLabel, list_choose_es: QListWidget, graph1: MplWidget, graph2: MplWidget, ir_results: QLabel):
    global sequence1, sequence2, dp, paths, es, edit_scripts, force_refresh_table, similarity_results

    def on_change(ind):
        global sequence1, sequence2, dp, paths, es, edit_scripts, force_refresh_table, similarity_results
        # print('CURRENT TAB', ind)

        if ind == 1:
            keys = list(similarity_results.keys())

            time_keys_methods = list(filter(lambda x: '_time' in x, keys))
            time_keys_conversions = list(filter(lambda x: 'pre_' in x, keys))
            # bins = len(time_keys)
            methods = []
            timing = []
            for k in time_keys_methods:
                full_method = k[:-5]
                ind = full_method.rfind('_')
                method_name = full_method[0:ind] if ind > 0 else full_method
                methods.append(method_name)
                timing.append(similarity_results[k])

            methods_conv = []
            timing_conv = []
            for k in time_keys_conversions:
                full_method = k[4:]
                print(full_method)
                method_name = full_method
                methods_conv.append(method_name)
                timing_conv.append(similarity_results[k])

            graph1.canvas.ax.cla()
            graph1.canvas.ax.bar(methods, timing, color='g', zorder=3, width=0.3)
            graph1.canvas.ax.set_xticklabels(methods, rotation=45, rotation_mode="anchor", ha="right")
            graph1.canvas.ax.set(title='Similarity Measure', xlabel='Method', ylabel='Time (ms)')
            graph1.canvas.ax.grid()
            graph1.canvas.draw()

            graph2.canvas.ax.cla()
            graph2.canvas.ax.bar(methods_conv, timing_conv, color='g', zorder=3, width=0.2)
            graph2.canvas.ax.set_xticklabels(methods_conv, rotation=45, rotation_mode="anchor", ha="right")
            graph2.canvas.ax.set(title='Preprocessing Steps', xlabel='Method', ylabel='Time (ms)')
            graph2.canvas.ax.grid()
            graph2.canvas.draw()

            results_keys = list(filter(lambda x: '_time' not in x and 'pre_' not in x, keys))
            final_label = '<html>'
            for k in results_keys:
                final_label = final_label + '<b>' + k + '</b>: \t' + str(round(similarity_results[k], 3)) + '<br/>'
            final_label = final_label + '</html>'
            ir_results_label.setStyleSheet('line-height:20')
            ir_results_label.setText(final_label)

        if ind == 2:
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
                    label_comparison.show()
                    label_comparison_title.show()
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
                line_edit_patch.setText(sequence1)
                button_start_patch.setEnabled(True)
            if edit_scripts != []:
                line_edit_patch.setText(sequence1)
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


def onInputNextClicked(eText1, eText2, tabs: QTabWidget, enable_wf: QCheckBox, set_methods: CheckableComboBox,
                       multiset_methods: CheckableComboBox, vector_methods: CheckableComboBox, combo_tf_idf: QComboBox):
    global sequence1, sequence2, force_refresh_table, similarity_results

    def on_click():
        global sequence1, sequence2, force_refresh_table, similarity_results
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
            # print('Inputs and costs confirmed:')
            # print(f'Seq1: {sequence1}. Seq2: {sequence2}')

            should_calculate_wf = enable_wf.isChecked()
            wanted_sets = set_methods.check_items()
            wanted_multiset = multiset_methods.check_items()
            wanted_vector = vector_methods.check_items()


            if should_calculate_wf or len(wanted_sets)+len(wanted_multiset)+len(wanted_vector) > 0:
                similarity_results = {}
                job_list = []

                if should_calculate_wf:
                    start = time.time()
                    similarity_results['wf'] = wf_score(sequence1, sequence2)  # TODO: User COSTS
                    end = time.time()
                    similarity_results['wf_time'] = (end-start)*1000
                    similarity_results['pre_wf'] = 0

                if len(wanted_sets) > 0:
                    job_list = []
                    start = time.time()
                    seq1_set = convert_to_set(sequence1)
                    seq2_set = convert_to_set(sequence2)
                    end = time.time()
                    for j in wanted_sets:
                        job_list.append(set_methods_names[set_selections.index(j)])

                    similarity_results.update(create_and_start_threads(job_list, seq1_set, seq2_set))
                    similarity_results['pre_set'] = (end-start)*1000
                    # print(result_set)

                if len(wanted_vector) > 0:
                    job_list = []
                    start = time.time()
                    wanted_type = combo_tf_idf.currentText().lower()
                    if wanted_type == 'tf':
                        seq1_vec = convert_to_tf_vector(sequence1)
                        seq2_vec = convert_to_tf_vector(sequence2)
                    elif wanted_type == 'idf':
                        seq1_vec = convert_to_idf_vector(sequence1, list_of_docs=[sequence1, sequence2])
                        seq2_vec = convert_to_idf_vector(sequence2, list_of_docs=[sequence1, sequence2])
                    else:
                        seq1_vec = create_tf_idf_vector(sequence1, list_of_docs=[sequence1, sequence2])
                        seq2_vec = create_tf_idf_vector(sequence2, list_of_docs=[sequence1, sequence2])
                    end = time.time()
                    for j in wanted_vector:
                        job_list.append(vector_methods_names[vector_selections.index(j)])

                    similarity_results.update(create_and_start_threads(job_list, seq1_vec, seq2_vec))
                    similarity_results['pre_vector'] = (end-start)*1000

                    # print(result_vec)

                if len(wanted_multiset) > 0:
                    job_list = []
                    start = time.time()
                    seq1_vec = convert_to_multi_set(sequence1)
                    seq2_vec = convert_to_multi_set(sequence2)
                    end = time.time()
                    for j in wanted_multiset:
                        job_list.append(multiset_methods_names[multiset_selections.index(j)])

                    similarity_results.update(create_and_start_threads(job_list, seq1_vec, seq2_vec))
                    similarity_results['pre_multiset'] = (end-start)*1000
                    # print(result_multi)


                print(wanted_sets)
                print(wanted_multiset)
                print(wanted_vector)
                print(similarity_results)
                force_refresh_table = True
                tabs.setTabEnabled(1, True)
                if should_calculate_wf:
                    tabs.setTabEnabled(2, True)
                tabs.setCurrentIndex(1)

    return on_click

def calc_prec_rec_f(nbr_returned, nbr_relevant, relevant_returned):
    TP = relevant_returned
    FP = nbr_returned - relevant_returned
    FN = nbr_relevant - relevant_returned

    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f_value = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    return prec, rec, f_value

def on_click_search(set_methods: CheckableComboBox, multiset_methods: CheckableComboBox, vector_methods: CheckableComboBox,
                    search_combo_tf_idf: QComboBox, query_edit: QLineEdit, k_edit: QLineEdit, results_list: QListWidget,
                    prec: QLabel, rec: QLabel, f: QLabel, nbr_relevant: QLineEdit, threshold_range: QLineEdit):
    global collection

    def on_click():
        global collection
        c = collection.count({})
        # should_calculate_wf = enable_wf.isChecked()
        query = query_edit.text()
        wanted_sets = set_methods.check_items()
        wanted_multiset = multiset_methods.check_items()
        wanted_vector = vector_methods.check_items()
        wanted_type = search_combo_tf_idf.currentText().lower()
        k = int(k_edit.text()) if k_edit.text().isdigit() else c
        threshold = float(threshold_range.text()) if threshold_range.text().isdecimal() else 0.
        wf_rel = int(nbr_relevant.text()) if nbr_relevant.text().isdigit() else 5

        if len(query) > 0 and len(wanted_sets)+len(wanted_multiset)+len(wanted_vector) > 0:
            job_list = []

            if len(wanted_sets) > 0:
                for j in wanted_sets:
                    job_list.append(set_methods_names[set_selections.index(j)])

            if len(wanted_vector) > 0:

                for j in wanted_vector:
                    job_list.append(vector_methods_names[vector_selections.index(j)])

            if len(wanted_multiset) > 0:
                for j in wanted_multiset:
                    job_list.append(multiset_methods_names[multiset_selections.index(j)])

            s = time.time()
            sorted_results_final = []

            def show_results(final_results):
                print('showing results at ' + str(time.time() - s))
                sorted_results = sorted(final_results, key=itemgetter(1), reverse=True)
                print('k is: ', k)
                results_list.clear()
                for i in range(k):
                    # print(threshold)
                    # print(sorted_results[i][1])
                    # print(threshold)
                    if sorted_results[i][1] > threshold:
                        sorted_results_final.append(sorted_results[i][0])
                        results_list.addItem(sorted_results[i][0])
                print(sorted_results_final)

            def wf_done(wf_results):
                print('showing performance at ' + str(time.time() - s))
                # print(query)
                # print(sorted_results_final, len(sorted_results_final))
                sorted_wf = sorted(wf_results, key=itemgetter(1), reverse=True)
                print('wf_sorted', wf_rel, sorted_wf[0: wf_rel])
                count_relevant_ret = 0
                for i in sorted_results_final:
                    if i in [x[0] for x in sorted_wf[0: wf_rel]] or query in i:
                        count_relevant_ret += 1
                        # print('SHOWING COLOR')
                        results_list.item(sorted_results_final.index(i)).setBackground(QColor('#83fc9e'))

                p, r, f_val = calc_prec_rec_f(len(sorted_results_final), wf_rel, count_relevant_ret)
                # print(p,r,f_val)
                prec.setText('Precision: ' + str(round(p, 3)))
                rec.setText('Recall: ' + str(round(r, 3)))
                f.setText('F-Value: ' + str(round(f_val,3)))

            create_search_threads(job_list, query, wanted_type, collection, show_results, wf_done)


            # show_results(final_results_search)

            # print(final_results_search)
            # print(wf_results)

    return on_click


def on_goto_ed(q: QLineEdit, search_list: QListWidget, seq1: QLineEdit, seq2: QLineEdit, tabs: QTabWidget):

    def on_click():
        seq1.setText(q.text())
        seq2.setText(search_list.selectedItems()[0].text())
        tabs.setCurrentIndex(0)

    return on_click


def on_export_to_file():
    global es

    save_path_and_ext = QFileDialog.getSaveFileName(filter='JSON (*.json)')
    save_path = save_path_and_ext[0]
    if save_path != '':
        # print(save_path)
        obj_to_save = {'edit_script': es}

        with open(save_path, 'w') as f:
            json.dump(obj_to_save, f, indent=4)


def on_goto_patching(tabs: QTabWidget):
    def on_click():

        tabs.setCurrentIndex(2)

    return on_click


def on_import_patching(list_choose_es: QListWidget, es_label: QLineEdit):
    global sequence1, sequence2, es

    def on_click():
        global sequence1, sequence2, es
        open_path_ext = QFileDialog.getOpenFileName(filter='JSON (*.json)')
        open_path = open_path_ext[0]

        if open_path != '':
            with open(open_path, 'r') as f:
                data = json.load(f)
                # sequence1 = data['seq1']
                # sequence2 = data['seq2']
                es = data['edit_script']
                es_label.setText(str(format_edit_script(es)))

            # i = 0
            list_choose_es.clear()
            # for e in edit_scripts:
            #     if len(e) == 0:
            #         view = f'Edit Script #{i + 1}: {format_edit_script(e)} - Edit script is empty --> sequences are already homomorphic'
            #     else:
            #         view = f'Edit Script #{i + 1}: {format_edit_script(e)}'
            #     list_choose_es.addItem(view)
            #     i += 1

            # radio_1.setText(sequence1)
            # radio_2.setText(sequence2)

            # print('loaded JSON file')

    return on_click


def on_select_es(es_label: QLabel, start_patch: QPushButton):
    global sequence1, sequence2, es, edit_scripts, inverted, prog_inv

    def on_change(ind):
        global sequence1, sequence2, es, edit_scripts, inverted, prog_inv

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


    return on_change


def on_reverse(label_new_es: QLabel):
    global es

    def on_click():
        global es
        es = generate_rev_es(es)
        label_new_es.setText(str(format_edit_script(es)))

    return on_click


def on_start_patching(line_edit_patch: QLineEdit, label_out: QLabel):
    global es

    def on_click():
        global es
        val = line_edit_patch.text()

        if len(val) > 0 and es != []:
            # print('PATCHING')
            error_code, patched = patching(es, val)
            if error_code == -1:
                showdialog('Error: The input sequence is not compatible with this edit script! It has a shorter sequence than necessary', is_error=True)
            elif error_code == 1:
                showdialog('Warning: The input sequence does not correspond to the one that generated this edit script. The result may not be valid.')

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
    loader.registerCustomWidget(CheckableComboBox)
    loader.registerCustomWidget(MplWidget)
    window = loader.load(ui_file)
    ui_file.close()

    # load table window
    ui_table_fil = QFile(ui_table_name)
    if not ui_table_fil.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)

    load_table = QUiLoader()
    # table_window = uic.loadUi(ui_table_fil)
    table_window = loader.load(ui_table_fil)
    ui_table_fil.close()

    ui_dataset_file = QFile(ui_dataset_name)
    if not ui_dataset_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)

    load_dataset_window = QUiLoader()
    dataset_window = loader.load(ui_dataset_file)
    ui_dataset_file.close()

    ui_xml_file = QFile(ui_xml_name)
    if not ui_xml_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_xml_file.errorString()}")
        sys.exit(-1)

    load_xml_window = QUiLoader()
    xml_window = loader.load(ui_xml_file)
    ui_xml_file.close()

    if not window:
        print(loader.errorString())
        sys.exit(-1)
    window.show()

    tab_widget = window.findChild(QTabWidget, 'tabWidget')
    tab_widget.setTabEnabled(1, False)
    tab_widget.setTabEnabled(2, False)
    # main window variables:
    # tab 1:
    next_button = window.findChild(QPushButton, 'button_input_next')
    btn_import_1 = window.findChild(QPushButton, 'btn_import_1')
    btn_import_2 = window.findChild(QPushButton, 'btn_import_2')

    btn_xml_import_1 = window.findChild(QPushButton, 'btn_xml_import_1')
    btn_xml_import_2 = window.findChild(QPushButton, 'btn_xml_import_2')

    edit_seq1 = window.findChild(QLineEdit, 'edit_sequence1')
    edit_seq2 = window.findChild(QLineEdit, 'edit_sequence2')

    wagner_checkbox = window.findChild(QCheckBox, 'wagner_checkbox')
    set_combo = window.findChild(CheckableComboBox, 'set_combo')
    multiset_combo = window.findChild(CheckableComboBox, 'multiset_combo')
    vector_combo = window.findChild(CheckableComboBox, 'vector_combo')
    combo_tf_idf = window.findChild(QComboBox, 'combo_tf_idf')

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

    # xml import window
    btn_browse_xml = xml_window.findChild(QPushButton, 'btn_xml_browse')
    list_dataset_xml = xml_window.findChild(QListWidget, 'list_data')
    btn_import_data_xml = xml_window.findChild(QPushButton, 'btn_import_data')

    # tab 2: performance
    graph1 = window.findChild(MplWidget, 'graph1')
    graph2 = window.findChild(MplWidget, 'graph2')
    ir_results_label = window.findChild(QLabel, 'ir_results_label')

    # tab 2: ED
    ed_matrix_table = window.findChild(QTableWidget, 'ed_matrix_table')
    label_cost = window.findChild(QLabel, 'label_ec')
    label_sim = window.findChild(QLabel, 'label_sim')
    es_list = window.findChild(QListWidget, 'es_list')
    label_title = window.findChild(QLabel, 'label_calculator')
    btn_to_patching = window.findChild(QPushButton, 'btn_next')
    btn_export_patching = window.findChild(QPushButton, 'btn_export')
    label_comparison_title = window.findChild(QLabel, 'label_comparison_title')
    label_comparison = window.findChild(QLabel, 'label_comparison')

    # search tab:
    edit_search_query = window.findChild(QLineEdit, 'edit_search_query')
    combo_tf_idf_search = window.findChild(QComboBox, 'combo_tf_idf_search')
    multiset_combo_search = window.findChild(CheckableComboBox, 'multiset_combo_search')
    set_combo_search = window.findChild(CheckableComboBox, 'set_combo_search')
    vector_combo_search = window.findChild(CheckableComboBox, 'vector_combo_search')
    edit_k = window.findChild(QLineEdit, 'edit_k')
    button_start_search = window.findChild(QPushButton, 'button_start_search')
    list_search_results = window.findChild(QListWidget, 'list_search_results')

    label_precision = window.findChild(QLabel, 'label_precision')
    label_recall = window.findChild(QLabel, 'label_recall')
    label_f_value = window.findChild(QLabel, 'label_f_value')
    edit_nbr_relevant = window.findChild(QLineEdit, 'edit_nbr_relevant')
    edit_threshold = window.findChild(QLineEdit, 'edit_threshold')
    btn_search_ed = window.findChild(QPushButton, 'btn_search_ed')


    # tab 3:
    btn_import = window.findChild(QPushButton, 'btn_import')
    list_patch_choose = window.findChild(QListWidget, 'list_patch_es')
    label_es_chosen = window.findChild(QLabel, 'label_es_chosen')
    # edit_patch_input = window.findChild(QLineEdit, 'edit_patch_input')
    # radio_patch_seq1 = window.findChild(QRadioButton, 'radio_patch_sequence1')
    # radio_patch_seq2 = window.findChild(QRadioButton, 'radio_patch_sequence2')
    line_edit_patch_sequence = window.findChild(QLineEdit, 'line_edit_patch_sequence')
    btn_patch = window.findChild(QPushButton, 'btn_patch')
    label_patched = window.findChild(QLabel, 'label_patched')
    btn_rev = window.findChild(QPushButton, 'btn_reverse')

    edit_cost_ins.setText(str(user_costs['insert']))
    edit_cost_del.setText(str(user_costs['delete']))

    combo_selections = ['Please Select Nucleotide...', *nucleotides]
    combo_from.insertItems(0, combo_selections)

    combo_tf_idf_selections = ['TF', 'IDF', 'TF-IDF']
    combo_tf_idf.addItems(combo_tf_idf_selections)

    set_combo.addItems(set_selections)
    multiset_combo.addItems(multiset_selections)
    vector_combo.addItems(vector_selections)

    combo_tf_idf_search.addItems(combo_tf_idf_selections)
    multiset_combo_search.addItems(multiset_selections[1:])
    set_combo_search.addItems(set_selections[1:])
    vector_combo_search.addItems(vector_selections)

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

    btn_browse_xml.clicked.connect(on_browse_xml(list_dataset_xml))
    btn_xml_import_1.clicked.connect(import_from_xml(1, xml_window, list_dataset_xml, btn_import_data_xml, edit_seq1, edit_seq2))
    btn_xml_import_2.clicked.connect(import_from_xml(2, xml_window, list_dataset_xml, btn_import_data_xml, edit_seq1, edit_seq2))

    edit_seq1.textEdited.connect(onSequence1Changed(edit_seq1))
    edit_seq2.textEdited.connect(onSequence2Changed(edit_seq2))
    next_button.clicked.connect(onInputNextClicked(edit_seq1, edit_seq2, tab_widget, wagner_checkbox, set_combo, multiset_combo, vector_combo, combo_tf_idf))
    tab_widget.currentChanged.connect(
        onTabChanged(radio_cost_user, ed_matrix_table, label_cost, label_sim, es_list, label_title,
                     label_es_chosen, line_edit_patch_sequence,
                     btn_patch, label_comparison, label_comparison_title, list_patch_choose, graph1, graph2, ir_results_label))
    edit_cost_ins.textEdited.connect(on_insert_cost_change('insert'))
    edit_cost_del.textEdited.connect(on_insert_cost_change('delete'))
    combo_from.currentIndexChanged.connect(on_combo_changed(cost_table))
    cost_table.cellChanged.connect(on_table_edited(combo_from, cost_table))

    # search tab:
    button_start_search.clicked.connect(on_click_search(set_combo_search, multiset_combo_search, vector_combo_search,
                                                        combo_tf_idf_search, edit_search_query, edit_k, list_search_results,
                                                        label_precision,label_recall,label_f_value, edit_nbr_relevant,
                                                        edit_threshold))

    btn_search_ed.clicked.connect(on_goto_ed(edit_search_query, list_search_results, edit_seq1, edit_seq2, tab_widget))

    # tab 2:
    es_list.currentRowChanged.connect(
        on_es_list_changed(ed_matrix_table, btn_to_patching, btn_export_patching))
    btn_to_patching.clicked.connect(on_goto_patching(tab_widget))

    # tab 3:
    btn_patch.clicked.connect(on_start_patching(line_edit_patch_sequence, label_patched))
    btn_export_patching.clicked.connect(on_export_to_file)
    btn_import.clicked.connect(on_import_patching(list_patch_choose, label_es_chosen))
    list_patch_choose.currentRowChanged.connect(on_select_es(label_es_chosen, btn_patch))

    btn_rev.clicked.connect(on_reverse(label_es_chosen))

    sys.exit(app.exec_())
