Skrypty bounding_box_train.py, bounding_box_create_labels.py, bounding_box_detect_angle.py służą odpowiednio do trenowania modelu, przewidywania ramek, wykrywania kąta obrotu obiektów

Wszystkie uruchamia się podając pliki konfiguracyjne jako argument wykonania (python3 bounding_box_train.py config_train.py). Argumenty konfiguracji są opisane w plikach konfiguracyjnych.

Architektura sieci jest zdefiniowana w pliku init_model.py, konfiguracja treningu w pliku lightning_module.py.


W katalogu tools znajdują się różne przydatne narzędzia, w tym skrypt create_location_data_from_labels.py który tworzy w podanym katalogu z danymi w formacie YOLO w katalogach images i labels katalog z plikami z przybliżonymi ramkami, konwertując dowolny zbiór danych w formacie YOLO na kompatybilny z siecią.

