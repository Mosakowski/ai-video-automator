# Raport techniczny projektu video_automator (stan na 2026-04-27)

## 1. Stack technologiczny i infrastruktura

- Jezyk: Python 3.x.
- Frontend: Streamlit (UI w app.py, single-page dashboard, custom CSS).
- Backend / silnik wideo: MoviePy + OpenCV + Pillow + CairoSVG.
- Analiza obrazu: Ultralytics YOLOv8 (model lokalny yolov8n.pt).
- Brak bazy danych w MVP; stan przechowywany w pamieci sesji i na dysku w folderze temp.
- Hosting: aktualnie lokalne uruchomienie (brak warstwy serwerowej, brak kontenerow); blueprint wskazuje przyszla migracje na rendering w chmurze (AWS/GCP workers).
- Zaleznosci systemowe: na Windows wymagany GTK3 Runtime (dla CairoSVG).

## 2. Pipeline generowania wideo (krok po kroku)

### 2.1 Wejscie danych
1) Uzytkownik wgrywa obrazy i/lub wideo (JPG/PNG/MP4/MOV) i opcjonalny plik audio (MP3).
2) Uzytkownik ustawia kolejnosc mediow (drag & drop w UI).
3) Uzytkownik konfiguruje watermark (logo), pozycje, skale i przezroczystosc.
4) Uzytkownik konfiguruje naglowek (tekst, styl, animacje, kolory, font, pozycje Grid/XY, opcjonalny Custom SVG).
5) UI generuje podglad statyczny (render_unified_mockup). To jest obraz PIL 540x960 z nalozonym logo i naglowkiem.

### 2.2 Przetwarzanie timeline
6) Pliki sa zapisywane do folderu temp (app.py -> cleanup_temp_dir + zapis binarny).
7) Silnik process_video_pipeline rozdziela wejscia na obrazy i wideo.
8) Wideo:
   - Pobranie pierwszej klatki, utworzenie rozmytego tla 9:16.
   - Dopasowanie wideo do 1080x1920 z zachowaniem proporcji.
   - Wyciszenie lub miks audio tła zgodnie z video_bg_volume.
9) Obrazy:
   - Detekcja obiektu (YOLOv8) i wyznaczenie centrum kadru.
   - Ken Burns z 3x oversampling, aby uniknac jittera.
10) Miedzy klipami stosowany crossfade (0.4 s).

### 2.3 Audio i synchronizacja
11) Jesli jest voiceover, dlugosc slajdow jest dopasowana do czasu audio.
12) Jesli finalny video jest dluzszy niz audio, jest przycinany.
13) Miks audio: istniejace audio wideo + voiceover w CompositeAudioClip.

### 2.4 Warstwy i render
14) Watermark: PNG z alfanumerycznym alfa przez OpenCV, nalozony przez MoviePy.
15) Dynamiczny naglowek:
   - Generacja SVG (style 1-8 lub Custom SVG) w generate_dynamic_header_img.
   - Render SVG do PNG przez CairoSVG, wyliczenie optycznego bounding box.
   - Pozycjonowanie przez resolve_header_top_left (GRID lub XY).
   - Opcjonalna animacja (slide-in / pop-up) przez funkcje easing.
16) Kompozycja warstw w jednym CompositeVideoClip.
17) Render finalny przez MoviePy write_videofile (libx264 + AAC, preset superfast, threads = cpu_count).

## 3. Zaleznosci zewnetrzne i API

### 3.1 Zaleznosci w requirements.txt
- streamlit==1.36.0
- ultralytics (YOLOv8)
- moviepy
- opencv-python-headless
- numpy
- pillow
- streamlit-drawable-canvas
- streamlit-sortables
- cairosvg==2.7.1

### 3.2 Integracje API
- Aktualnie brak bezposrednich integracji z zewnetrznymi API (OpenAI, ElevenLabs itp.).
- Blueprint przewiduje integracje w kolejnych etapach:
  - TTS: ElevenLabs / OpenAI.
  - Transkrypcja: Whisper.
  - Publikacja: TikTok / Instagram / YouTube (planowane).

## 4. Obecny stan projektu (MVP/WIP/Koncepcja)

### MVP - zaimplementowane
- Streamlit UI z uploadem mediow, konfiguracja logo i naglowka.
- Render pipeline dla obrazow i wideo (MoviePy + OpenCV).
- Ken Burns z YOLO do centrowania kadru.
- Dynamiczne naglowki (wiele stylow + Custom SVG) przez CairoSVG.
- Podglad statyczny generowany lokalnie (PIL).

### WIP / czesciowo zaimplementowane
- Animacje naglowka (intro/outro) - dzialaja, ale bez edytora timeline.
- Mieszanie audio i dopasowanie czasu slajdow do voiceover.

### Koncepcja / planowane
- Automatyczne napisy (transkrypcja).
- Integracja TTS (ElevenLabs/OpenAI).
- Cloud rendering (worker pool).
- Biblioteka szablonow i zapisywanie presetow.
- Integracje z platformami publikacji.

## 5. Architektura danych i stan aplikacji

- Brak DB, brak persistent storage w MVP.
- Stan: dane sa w pamieci sesji Streamlit i na dysku w katalogu temp.
- Media i pliki posrednie:
  - Wszystkie uploady zapisywane do temp jako pliki tymczasowe.
  - Wynikowy render zapisany jako output.mp4.
- Dane konfiguracji generowane bezposrednio z UI i przekazywane funkcjami (brak modelu ORM).

## 6. Glowne wyzwania techniczne (pain points)

1) Wydajnosc renderu MoviePy (CPU, RAM). Ryzyko leakow pamieci na dluzszych renderach.
2) Czas generowania (YOLO + Ken Burns + render x264) - duze obciazenie CPU.
3) Brak persistent storage i brak kolejkowania zadan - brak skalowania horyzontalnego.
4) Zaleznosc od GTK3 Runtime na Windows (CairoSVG) - potencjalne problemy instalacyjne.
5) Synchronizacja audio-wideo w przypadku mieszanych timeline (obrazy + wideo) wymaga ostroznego zarzadzania czasem slajdow.
6) Brak asynchronicznosci / workerow - UI blokuje sie w trakcie renderu.
7) Ograniczona kontrola nad bledami w plikach wejsciowych (uszkodzone wideo, nietypowe kodeki).

## 7. Rekomendowane nastepne kroki dla doradcy

- Zaproponowac architekture async: kolejka (np. Celery/RQ) + storage (S3/Blob) + worker renderujacy.
- Rozbic pipeline na etapy i zapisac intermediate assets (cache klipow).
- Zdefiniowac kontrakt API dla przyszlego frontendu (np. REST/GraphQL).
- Zaplanowac integracje TTS i transkrypcji z limitem kosztowym i retry policy.
- Okreslic docelowy hosting i metryki (czas renderu, koszt CPU, RAM, rozmiar plikow).
