#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# ファイル名を指定
filename = '出勤打刻画面ログイン.txt'

# ファイルを読み込む
with open(filename, 'r') as file:
    line = file.readline().strip()

# セミコロンで分割
username, password = line.split(';')

# ブラウザのドライバーを設定
driver = webdriver.Edge()

# Webサイトにアクセス
driver.get("http://jnjhp.in.kyocera.co.jp/fw/dfw/HRAAM/hrweb/common/empPortal")
#ウインドウ最大化
driver.maximize_window()

# ユーザー名とパスワードを入力
username_ = driver.find_element(By.ID, "username")
password_ = driver.find_element(By.ID, "password")
username_.send_keys(username)
password_.send_keys(password)

# ログインボタンをクリック
login_button = driver.find_element(By.ID, "submit")
login_button.click()

# 必要に応じて、他の操作を追加

login_button = driver.find_element(By.ID, "dakokuLogin")
login_button.click()

#login_button = driver.find_element(By.ID, "shukkin")
#login_button.click()

