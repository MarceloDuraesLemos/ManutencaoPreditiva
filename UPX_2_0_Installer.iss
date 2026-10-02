#define MyAppName "UPX 2.0"
#define MyAppVersion "2.0"
#define MyAppPublisher "UPX 2.0"
#define MyAppExeName "UPX_2_0.exe"

[Setup]
AppId={{A1B8A563-58E8-4D64-93E7-9E8C43C0D202}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

DefaultDirName={localappdata}\Programs\UPX 2.0
DefaultGroupName=UPX 2.0
DisableProgramGroupPage=yes

PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

OutputDir=instalador
OutputBaseFilename=UPX_2_0_Setup

Compression=lzma2
SolidCompression=yes

WizardStyle=modern
SetupLogging=yes

UninstallDisplayName=UPX 2.0
UninstallDisplayIcon={app}\{#MyAppExeName}

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar um atalho na Área de Trabalho"; GroupDescription: "Atalhos adicionais:"

[Files]
Source: "dist\UPX_2_0\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\UPX 2.0"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\UPX 2.0"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Executar UPX 2.0"; Flags: nowait postinstall skipifsilent