#!/usr/bin/env python
# coding: utf-8

# In[9]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# ブラウザのドライバーを設定
driver = webdriver.Edge()

# Webサイトにアクセス
driver.get("https://kyoceragp.sharepoint.com/sites/FCkaihatsu_yokaichi/Shared%20Documents/Forms/AllItems.aspx?viewid=666d9991%2Df98f%2D4a45%2D809c%2Dee5b23bded9a")

# ボタンをクリック
button = driver.find_element(By.XPATH, '//button[@type="button" and @role="menuitem" and contains(@class, "ms-Button--commandBar") and @aria-label="その他"]')
button.click()

button = driver.find_element(By.XPATH, '//li[@role="presentation" and @title="OneDrive 内のこのフォルダーへショートカットを追加する"]//button[@title="OneDrive 内のこのフォルダーへショートカットを追加する" and @name="OneDrive へのショートカットの追加" and @data-automationid="addShortcutFromTeamSiteCommand"]')
button.click()

# 待機
time.sleep(5)
driver.quit()

# 待機
#time.sleep(120)

import tkinter as tk
from tkinter import messagebox
import shutil
import os

dire=os.getcwd()
dire=dire.split(os.sep)
result = dire[2]

def get_input():
    # Tkinterのルートウィンドウを作成
    root = tk.Tk()
    root.title("入力と選択")

    # 選択肢を作成
    tk.Label(root, text="　　　　所属課を選んでください:　　　　").pack()
    option_var = tk.StringVar(root)
    option_var.set("先行開発課")  # デフォルト値を設定

    options = ["先行開発課", "東近江プロセス開発課", "生産技術開発課"]
    option_menu = tk.OptionMenu(root, option_var, *options)
    option_menu.pack()

    # 確認ボタンのコールバック関数
    def on_confirm():
        selected_option = option_var.get()
        confirmation = messagebox.askyesno("確認", f"選択された値: {selected_option}\nこれでよろしいですか？")
        if confirmation:
            with open("所属課.txt", "w") as file:
                file.write(f"{selected_option}")
            print(f"選択された値: {selected_option}")
            
            root.quit()

    # 確認ボタンを作成
    confirm_button = tk.Button(root, text="確認", command=on_confirm)
    confirm_button.pack()

    root.mainloop()

    # Tkinterのルートウィンドウを破棄
    root.destroy()

# 関数を呼び出して入力を取得
get_input()



def copy_files(source_folder, destination_folder):
    # ソースフォルダが存在するか確認
    if not os.path.exists(source_folder):
        print(f"ソースフォルダ '{source_folder}' が存在しません。")
        return
    
    # 目的フォルダが存在しない場合は作成
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 目的フォルダ内のすべてのファイルを削除
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"削除しました: {file_path}")
        except Exception as e:
            print(f"削除に失敗しました {file_path}. 理由: {e}")
    
    # ソースフォルダ内のすべてのファイルをコピー
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # ファイルをコピー
        shutil.copy2(source_file, destination_file)
        print(f"コピーしました: {source_file} から {destination_file} へ")
    
# 使用例
with open('所属課.txt', 'r') as file:
    dev = file.read()
source_folder = r"C:\Users\123456\OneDrive - 京セラ株式会社\ドキュメント - FC技術開発部 八日市\保護具点検用\課\アプリ"
source_folder = source_folder.replace('123456', result)
source_folder = source_folder.replace('課', dev)
destination_folder = r"C:\Users\123456\path_to_destination_folder"
destination_folder = destination_folder.replace('123456', result)

copy_files(source_folder, destination_folder)




import win32com.client

def create_shortcut(folder_path):
    # フォルダ内のすべてのファイルを取得
    files = os.listdir(folder_path)
    
    # .exeファイルをフィルタリング
    exe_files = [file for file in files if file.endswith('.exe')]
    
    # 各.exeファイルのショートカットを作成
    for exe_file in exe_files:
        # .exeファイルのパスを定義
        exe_path = os.path.join(folder_path, exe_file)
        
        # ショートカットのパスを定義
        shortcut_path = os.path.join("C:\\Users\\123456\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup", f"{os.path.splitext(exe_file)[0]}.lnk")
        shortcut_path = shortcut_path.replace('123456', result)
        
        # ショートカットを作成
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = exe_path
        shortcut.WorkingDirectory = folder_path  # 作業フォルダーを設定
        shortcut.save()

# フォルダパスを指定
folder_path = r"C:\Users\123456\path_to_destination_folder"
folder_path = folder_path.replace('123456', result)

# 指定フォルダ内のすべての.exeファイルのショートカットを作成
create_shortcut(folder_path)


# In[ ]:




