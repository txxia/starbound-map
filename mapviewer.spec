# -*- mode: python -*-
import os

spec_root = os.path.abspath('.')
block_cipher = None


a = Analysis(['mapviewer.py'],
             pathex=[spec_root],
             binaries=[],
             datas=[('glfw3.dll', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          Tree('assets'),
          a.zipfiles,
          a.datas,
          name='StarboundMapViewer',
          version='version.txt',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )
