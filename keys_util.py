import win32api as wapi
import win32con as wc

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.';$/\\":
    keyList.append(char)

def key_check():
    keys = []
    if wapi.GetAsyncKeyState(wc.VK_LEFT) != 0: 
        keys.append('A')
    if wapi.GetAsyncKeyState(wc.VK_RIGHT) != 0: 
        keys.append('D')
    if wapi.GetAsyncKeyState(wc.VK_UP) != 0: 
        keys.append('W')
    if wapi.GetAsyncKeyState(wc.VK_DOWN) != 0: 
        keys.append('S')
    return keys