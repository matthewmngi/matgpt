bigram: I'tinyellotharondelink a sid wis eeanfiet ave ong Yong a ur _
I ht tearn I asso plllere ikangiencat
loss:

bigram + single head attention:
Hew asuto ntoifinjul aray . I bere hre you pit I's . Shain ? ’ s arcal yome . Che wre to . San ’ rls
{'train': tensor(2.1922), 'val': tensor(2.1933)}

bigram + multi head attention: 
How ’ s mut bot'm ariJupea beany omand werlto yopme plsotye tabeplakes ? Ye'lt . O rugamely luply !
{'train': tensor(2.0190), 'val': tensor(2.0096)}

bigram + mha + feed fwd:
Whiand chat whave . Yet inesto hea lailte donak . Helats . Im bout , wak ? Ah sbse epet . Mam wer . 
{'train': tensor(2.0416), 'val': tensor(2.0449)}

bigram + mha + feed fwd + block architecture + residual connections:
Dion . He oway , befuth you thindwiDnice at foul offeer's realy . How you faates with any you fough
{'train': tensor(1.8179), 'val': tensor(1.8234)}

bigram + mha + feed fwd + block architecture + residual connections + layer norm:
You a st the cromelime ? Yes , and theand fodidind in complatile gen the toces you my hankink the yo
{'train': tensor(1.8061), 'val': tensor(1.8066)}