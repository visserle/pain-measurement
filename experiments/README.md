2023.12.26
- fragebogen für pilotierung erstellen
  	- zu lang / zu kurz
	- abgeschaltet?
    - strategien gegen schmerz?
    - tempverlauf nachzeichnen
    - waren es die werte der kalibrierung
    - ...
- dryelektroden anfragen imotions
- add send temp to imotions even after stimuli?

2023.12.21
- mit Anderas über EEG gucken
- erwähnen, dass weniger geld ausgezahlt wird, wenn nicht kontinuierlich gerated wird?
- ~~imotions marker verbessern, siehe manual~~

2023.12.19 (Björn)
- ~~schreibfehler noch mit drin? how?~~
- instruktion für thermode: fest sitzen aber ohne druck, oder dass blut abgeschnürt wird, probanden fragen
- an der mittleren haut-stelle kalibrieren
- erst flächen auftragen, damit man weiß wo die creme genau hin muss
- Handschuhe für Creme
- schmerz als "neue qualität" zusätzlich uzm capsicin, stechen als mögliches schlagwort
- wichtig, dass wir konsistente werte finden
- vas100 neu definieren, bild anpassen 10 = nicht akzeptable im rahmen des Experiemnts
- kalibrierung vas delta check für min 1 ° abstand zwixchen vas0 und 80
- kalibrierung ende sagen was wir gemacht haben, teil 1 und teil2 als minimum und maximum für folgendes experiemnt, text mit werte aus hauptexpierment streichen
- Maus defokussiert??? -° versuch mit time.sleep zu fixen
- alles gesundheitlich unbedenktlich, auch wenn max. temp über das gsamte exp.
- ~~marker einbauen, den probanden senden können, wenn die sie letzten paar sekunden nicht aufmerksam geratet haben, mittlere scroll taste~~ -> möglich, aber erstmal ohne probieren, überforderung, etc.
- stuhl mit armlehnen?
- imotions aufname beenden, wenn psychopy gestoppt wird (mit marker für abort)
- generell imotions mkr erüberarbeiten, merker für stimuli, etc. (am besten auch mit seed information)
- in einladungsmail vermerk zu kleidung
- keine Brille
- eda problem -> heart monitor aktiviert, aber nicht vorhanden. muss aus.


2023.12.18
- batterie für bluetooth tastatur
- ~~stimluli functions are of different lenhtgs~~
- wie aus mms daten sichern?
- mms einstellen


2023.12.15
- ~~define mms programm for calibration and experiment, esp. the time windows, short for calibration, long for experiment~~


2023.12.07
- isoluminanz erneut adjustieren
- rope fragebogen marie JC
- ~~klemmbretter~~


2023.12.06
- zuweisung der schimmer-geräte war falsch -> ecg kann als eda nicht interpretiert werden, logisch
- eda mit elektroden an hand -> ext
- ra eda-elektroden verwenden (?), björn fragen
- einmalhandschuhe fürs eincremen bereitstellen
- große flasche desinfektionsmittel


2023.12.05
- Versuchspersonen sagen, dass Capsaicin nochmal abends brennen kann, wenn sie duschen/heißes Wasser


2023.11.30
- Psychopy Kreuz & Abb. seitlich verzerrt
- Aufgrund der Capsaicin Creme ...
- ~~Kalibrierte Werte =/= int~~
- EKG: Abbildung, Text, ...
- Zusammenfassung / Check-Liste -> 3 Seiten max.
- sinnvolle Reihenfolge finden
- Bluetooth Keyboars / Maus einrichten
- ~~Skripte schreiben, um Experimente zu starten~~
- Fragebögen
- ~~Plateu etvl kürzer?~~
- ~~EKG funktioniert nicht?~~
- ~~EMG funktioniert nicht?~~


2023.11.20
- Frage an Björn: Maximale RoR für pathways
    -> WICHTIG um die Länge zwischen den Triggern genau zu bestimmen
- ~~calibrierung parameter finden~~


2023.11.13
- ~~mms basline weiter nach unten setzen, evtl 28°C~~
- ~~ekg ausprobieren~~
- reihenfolge der seeds randomisieren
- ~~Info einbauen dass schmerz mit stechender komponente~~


2023.10.26
- ~~ende des trials muss zurück auf baseline (was ist baseline?)~~


2023.10.16
- ~~hautaufwärmen nach wechseln der hautstelle im hauptexperiment~~
- ~~viele details im sop~~


2023.09.29
- ~~isoluminante Farben für die Stimulation finden~~
- ~~an zwei Farbkreisen ausstesten, ob Unterschied gefunden werden kann in Pupillometrie~~
- ~~momentan als 3X big, 2 oder 1x small decrease~~
- ~~da die stimuli function nun mit ramp on endet, sollte in psychopy nicht sofort auf mms baselevel zurückgefahren werden, da sonst relief expectations entstehen~~


2023.09.25
- ~~ergänze warning calib 70 one direction~~
- streung vas70 estimator
- ~~back to baseline afer trial, maybe in execctc,~~
~~leerlaufafterprepctc~~
~~addinfotrialnumber~~
- ~~maybe vas70 first, vas0 second~~


2023.08.31
- ~~ramp off time should not be part of the stimilus length~~
-> ~~calculate time based on rate_of_rise and add it in every trial~~
- ~~default frame rate is 60,~~
~~but in the experiment the frame rate will often change to 30~~
-> ~~that's why nothing is dependent on frame counts, but on clocks / time~~

