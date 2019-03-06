import numpy as np

import win32api
import psutil
from ctypes import *
from ctypes.wintypes import *

class MODULEENTRY32(Structure):
    _fields_ = [( 'dwSize' , DWORD ) , 
                ( 'th32ModuleID' , DWORD ),
                ( 'th32ProcessID' , DWORD ),
                ( 'GlblcntUsage' , DWORD ),
                ( 'ProccntUsage' , DWORD ) ,
                ( 'modBaseAddr' , POINTER(BYTE) ) ,
                ( 'modBaseSize' , DWORD ) , 
                ( 'hModule' , HMODULE ) ,
                ( 'szModule' , c_char * 256 ),
                ( 'szExePath' , c_char * 260 ) ]

class MemoryReader():
    # Open the process for memory reading
    CreateToolhelp32Snapshot= windll.kernel32.CreateToolhelp32Snapshot
    Process32First = windll.kernel32.Process32First
    Process32Next = windll.kernel32.Process32Next
    Module32First = windll.kernel32.Module32First
    Module32Next = windll.kernel32.Module32Next
    GetLastError = windll.kernel32.GetLastError
    GetPriorityClass = windll.kernel32.GetPriorityClass
    OpenProcess = windll.kernel32.OpenProcess
    ReadProcessMemory = windll.kernel32.ReadProcessMemory
    CloseHandle = windll.kernel32.CloseHandle
        
    PROCESS_ALL_ACCESS = 0x1F0FFF
    TH32CS_SNAPMODULE = 0x00000008
    TH32CS_SNAPTHREAD = 0x00000004
    TH32CS_SNAPPROCESS = 2

    def __init__(self, process):
        self._pid = 0
        self._hwnd = 0
        self._windows = []
        self._process = process

        #Return a list of processes with the name
        ls = []
        for p in psutil.process_iter(attrs=['name']):
            if p.info['name'] == self._process:
                ls.append(p)

        self._pid = ls[0].pid

        # The base address of the process
        self._base_addr = win32api.GetModuleHandle(None)

        # The key that ends the session (P)
        self._end_session_key = 0x50

        # Open the process to read memory
        self._hProc = MemoryReader.OpenProcess(MemoryReader.PROCESS_ALL_ACCESS,
                                               False, self._pid)

    def get_pid(self):
        return self._pid

    def close_handle(self):
        MemoryReader.CloseHandle(self._hProc)

    def read_memory(self, address, val_type):
        val = val_type()

        buffer = c_char_p(b"The data goes here")
        bufferSize = sizeof(buffer)
        bytesRead = c_ulong(0)

        MemoryReader.ReadProcessMemory(self._hProc, address, buffer,
                                       bufferSize, byref(bytesRead))


        memmove(ctypes.byref(val), buffer, ctypes.sizeof(val))

        return val.value

    def get_address(self, address, val_type, offsets=[]):
        for offset in offsets:
            print(address)
            address = self.read_memory(address, c_byte)

            address += offset

        return self.read_memory(address, val_type)

    def end_session(self):
        return win32api.GetAsyncKeyState(self._end_session_key)
