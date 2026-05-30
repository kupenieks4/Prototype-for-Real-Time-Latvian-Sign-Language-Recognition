# Prototype-for-Real-Time-Latvian-Sign-Language-Recognition

Šajā repozitorijā ir ievietoti bakalaura darba laikā izstrādātie latviešu zīmju valodas reāllaika atpazīšanas prototipa faili. Tajā nav iekļautas oriģinālās, apgrieztās, video datu kopas. Tās netika pievienotas privātuma un failu izmēra dēļ, taču, tā vietā ir iekļauti no tām izgūtie MediaPipe rokas orientieru dati, kas tika izmantoti modeļa apmācībai un novērtēšanai.

Prototips izmanto MediaPipe rokas orientieru noteikšananai un Bi-LSTM neironu tīkla modeli, lai reāllaikā atpazītu latviešu zīmju valodas zīmes no datu kopas.

Galvenais fails reāllaika prototipa demonstrēšanai ir: 7_prototips.py

Pārējie faili ir iekļauti, lai demonstrētu izstrādes procesu un darba gaitā izmantotos piegājienus: datu sagatavošana, orientieru izgūšana, modeļa apmācība, modeļa novērtēšana, grupētā krusta validācija un reāllaika prototipa pārbaude.
