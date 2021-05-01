from PySide6.QtWidgets import QApplication, QComboBox
from PySide6.QtGui import QStandardItemModel
from PySide6.QtCore import Qt


class CheckableComboBox(QComboBox):
    def __init__(self, *args):
        super(CheckableComboBox, self).__init__(*args)
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(QStandardItemModel(self))

    def handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
            # print(item.text() + " was unselected.")
        else:
            item.setCheckState(Qt.Checked)
            # print(item.text() + " was selected.")
        self.check_items()

    def item_checked(self, index):
        item = self.model().item(index, 0)
        return item.checkState() == Qt.Checked

    def check_items(self):
        checkedItems = []
        for i in range(self.count()):
            if self.item_checked(i):
                checkedItems.append(self.model().item(i, 0).text())
        return checkedItems
