import pygame
import argparse
import os
import math

# --- Константы (значения по умолчанию) ---
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
FPS = 60
DEFAULT_TARGET_PIXEL_SIZE = 5
DEFAULT_STAGE1_ZOOM_SPEED = 1.014 # Скорость для Этапа 1 (масштабирование), > 1.0
DEFAULT_STAGE2_SPACING_SPEED = 1.014 # Скорость для Этапа 2 (расстояния), > 1.0

# Цвета
BLACK = (0, 0, 0)

# --- Глобальные переменные состояния ---
screen = None
clock = None
original_image = None
scaled_image_for_initial_display = None

is_animating = False
animation_stage = 0  # 0: не анимируется, 1: этап масштабирования, 2: этап расстояний
target_pixel_on_original_image_coords = None # (float, float) координаты на исходном изображении

# Для Этапа 1 (масштабирование)
zoom_scale_stage1 = 1.0

# Для Этапа 2 (расстояния)
current_pixel_spacing = 1.0
initial_pixel_spacing = 1.0 # Будет равно TARGET_PIXEL_SIZE

# Для сохранения кадров
save_frames_enabled = False
output_frames_dir = "frames"
frame_count = 0

# Настраиваемые через аргументы командной строки
TARGET_PIXEL_SIZE = DEFAULT_TARGET_PIXEL_SIZE
STAGE1_ZOOM_SPEED = DEFAULT_STAGE1_ZOOM_SPEED
STAGE2_SPACING_SPEED = DEFAULT_STAGE2_SPACING_SPEED
SCREEN_WIDTH = DEFAULT_SCREEN_WIDTH
SCREEN_HEIGHT = DEFAULT_SCREEN_HEIGHT


def parse_arguments():
    global output_frames_dir, save_frames_enabled, TARGET_PIXEL_SIZE
    global STAGE1_ZOOM_SPEED, STAGE2_SPACING_SPEED, SCREEN_WIDTH, SCREEN_HEIGHT
    
    parser = argparse.ArgumentParser(description="Двухэтапная анимация приближения изображения в Pygame.")
    parser.add_argument("image_path", help="Путь к входному PNG изображению.")
    parser.add_argument("--output_dir", default="frames", help="Папка для сохранения кадров анимации (по умолчанию: 'frames').")
    parser.add_argument("--save_frames", action="store_true", help="Включить сохранение кадров анимации.")
    parser.add_argument("--target_pixel_size", type=int, default=DEFAULT_TARGET_PIXEL_SIZE,
                        help=f"Размер 'пикселей' для Этапа 2 (по умолчанию: {DEFAULT_TARGET_PIXEL_SIZE}).")
    parser.add_argument("--stage1_speed", type=float, default=DEFAULT_STAGE1_ZOOM_SPEED,
                        help=f"Скорость масштабирования для Этапа 1 (множитель за кадр, > 1.0) (по умолчанию: {DEFAULT_STAGE1_ZOOM_SPEED}).")
    parser.add_argument("--stage2_speed", type=float, default=DEFAULT_STAGE2_SPACING_SPEED,
                        help=f"Скорость увеличения расстояний для Этапа 2 (множитель за кадр, > 1.0) (по умолчанию: {DEFAULT_STAGE2_SPACING_SPEED}).")
    
    args = parser.parse_args()

    output_frames_dir = args.output_dir
    save_frames_enabled = args.save_frames
    TARGET_PIXEL_SIZE = args.target_pixel_size
    STAGE1_ZOOM_SPEED = args.stage1_speed
    STAGE2_SPACING_SPEED = args.stage2_speed

    if TARGET_PIXEL_SIZE <= 0:
        print("Ошибка: target_pixel_size должен быть положительным числом.")
        exit(1)
    if STAGE1_ZOOM_SPEED <= 1.0:
        print("Ошибка: stage1_speed должен быть больше 1.0.")
        exit(1)
    if STAGE2_SPACING_SPEED <= 1.0:
        print("Ошибка: stage2_speed должен быть больше 1.0.")
        exit(1)

    return args.image_path

def init_pygame(image_path_arg):
    global screen, clock, original_image, scaled_image_for_initial_display, initial_pixel_spacing
    global save_frames_enabled, output_frames_dir
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Двухэтапная анимация приближения")
    clock = pygame.time.Clock()

    try:
        original_image = pygame.image.load(image_path_arg).convert_alpha()
    except pygame.error as e:
        print(f"Ошибка загрузки изображения: {image_path_arg} - {e}")
        pygame.quit()
        exit(1)

    scaled_image_for_initial_display = pygame.transform.scale(original_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    initial_pixel_spacing = float(TARGET_PIXEL_SIZE) # Это для Этапа 2

    reset_animation_state()

    if save_frames_enabled:
        if not os.path.exists(output_frames_dir):
            try:
                os.makedirs(output_frames_dir)
                print(f"Папка для кадров '{output_frames_dir}' создана.")
            except OSError as e:
                print(f"Не удалось создать папку '{output_frames_dir}': {e}")
                save_frames_enabled = False 
        else:
            print(f"Кадры будут сохраняться в '{output_frames_dir}'.")


def reset_animation_state():
    global is_animating, animation_stage, target_pixel_on_original_image_coords
    global current_pixel_spacing, frame_count, zoom_scale_stage1
    
    is_animating = False
    animation_stage = 0 # Нет анимации
    target_pixel_on_original_image_coords = None
    
    zoom_scale_stage1 = 1.0 # Сброс для Этапа 1
    current_pixel_spacing = initial_pixel_spacing # Сброс для Этапа 2
    
    frame_count = 0
    
    if scaled_image_for_initial_display:
        screen.blit(scaled_image_for_initial_display, (0, 0))
    else:
        screen.fill(BLACK)
    pygame.display.flip()


def start_animation(click_pos_screen):
    global is_animating, animation_stage, target_pixel_on_original_image_coords
    global frame_count, zoom_scale_stage1, current_pixel_spacing
    
    if not original_image: return

    original_img_width = original_image.get_width()
    original_img_height = original_image.get_height()

    img_center_x = original_img_width / 2.0
    img_center_y = original_img_height / 2.0


    ox = (img_center_x / SCREEN_WIDTH) * original_img_width
    oy = (img_center_y / SCREEN_HEIGHT) * original_img_height
    
    target_pixel_on_original_image_coords = (ox, oy)

    if save_frames_enabled:
        try:
            pygame.image.save(screen, os.path.join(output_frames_dir, f"frame_{frame_count:05d}.png"))
            print(f"Сохранен начальный кадр: frame_{frame_count:05d}.png (экран в момент клика)")
            frame_count += 1
        except pygame.error as e:
            print(f"Ошибка сохранения начального кадра (перед анимацией): {e}")

    is_animating = True
    animation_stage = 1 # Начинаем с Этапа 1
    zoom_scale_stage1 = 1.0 # Начальный масштаб для Этапа 1
    current_pixel_spacing = initial_pixel_spacing # Устанавливаем на случай прямого перехода к Этапу 2 (маловероятно)
    
    print(f"Начало анимации к пикселю ({ox:.2f}, {oy:.2f}). Этап 1 (масштабирование).")

def execute_stage2_drawing_logic():
    """Логика отрисовки для Этапа 2 (увеличение расстояний). Возвращает False, если этап завершен."""
    global current_pixel_spacing, frame_count

    screen.fill(BLACK)
    tx_orig, ty_orig = target_pixel_on_original_image_coords
    orig_w, orig_h = original_image.get_width(), original_image.get_height()
    screen_center_x = SCREEN_WIDTH / 2.0
    screen_center_y = SCREEN_HEIGHT / 2.0
    pixels_drawn_this_frame = 0

    if current_pixel_spacing <= 0: return False # Предосторожность

    max_orig_dist_x = (SCREEN_WIDTH / 2.0 + TARGET_PIXEL_SIZE / 2.0) / current_pixel_spacing
    max_orig_dist_y = (SCREEN_HEIGHT / 2.0 + TARGET_PIXEL_SIZE / 2.0) / current_pixel_spacing
    ox_start = math.floor(tx_orig - max_orig_dist_x)
    ox_end = math.ceil(tx_orig + max_orig_dist_x)
    oy_start = math.floor(ty_orig - max_orig_dist_y)
    oy_end = math.ceil(ty_orig + max_orig_dist_y)

    for oy_orig_int in range(oy_start, oy_end + 1):
        if not (0 <= oy_orig_int < orig_h): continue
        for ox_orig_int in range(ox_start, ox_end + 1):
            if not (0 <= ox_orig_int < orig_w): continue
            
            screen_x_center_of_block = screen_center_x + (ox_orig_int - tx_orig) * current_pixel_spacing
            screen_y_center_of_block = screen_center_y + (oy_orig_int - ty_orig) * current_pixel_spacing
            rect_left = screen_x_center_of_block - TARGET_PIXEL_SIZE / 2.0
            rect_top = screen_y_center_of_block - TARGET_PIXEL_SIZE / 2.0

            if rect_left < SCREEN_WIDTH and rect_top < SCREEN_HEIGHT and \
               rect_left + TARGET_PIXEL_SIZE > 0 and rect_top + TARGET_PIXEL_SIZE > 0:
                try:
                    color = original_image.get_at((ox_orig_int, oy_orig_int))
                    pygame.draw.rect(screen, color, (rect_left, rect_top, TARGET_PIXEL_SIZE, TARGET_PIXEL_SIZE))
                    pixels_drawn_this_frame += 1
                except IndexError: pass

    if save_frames_enabled:
        try:
            pygame.image.save(screen, os.path.join(output_frames_dir, f"frame_{frame_count:05d}.png"))
            frame_count += 1
        except pygame.error as e: print(f"Ошибка сохранения кадра (Этап 2): {e}")

    current_pixel_spacing *= STAGE2_SPACING_SPEED
    
    # Условия завершения Этапа 2
    if pixels_drawn_this_frame == 0 and current_pixel_spacing > TARGET_PIXEL_SIZE * 1.1 : # Было initial_pixel_spacing
        print(f"Этап 2 завершен: пиксели не отрисованы. Конечное расстояние: {current_pixel_spacing:.2f}")
        return False
    # Более строгая проверка на случай, если current_pixel_spacing очень большой
    if max_orig_dist_x < 0.5 and max_orig_dist_y < 0.5 and pixels_drawn_this_frame == 0:
        print(f"Этап 2 завершен: масштаб слишком велик. max_orig_dist: ({max_orig_dist_x:.3f}, {max_orig_dist_y:.3f}).")
        return False
    return True


def process_animation_frame():
    """Обработка одного кадра анимации, управляет этапами."""
    global is_animating, animation_stage, zoom_scale_stage1, current_pixel_spacing, frame_count

    if animation_stage == 1:
        # --- Этап 1: Плавное масштабирование ---
        screen.fill(BLACK)
        
        orig_w = float(original_image.get_width())
        orig_h = float(original_image.get_height())

        # Идеальная ширина/высота просматриваемой области в пикселях исходного изображения
        current_view_width_orig = orig_w / zoom_scale_stage1
        current_view_height_orig = orig_h / zoom_scale_stage1

        # Координаты верхнего левого угла просматриваемой области
        vp_x = target_pixel_on_original_image_coords[0] - current_view_width_orig / 2.0
        vp_y = target_pixel_on_original_image_coords[1] - current_view_height_orig / 2.0

        # Прямоугольник для subsurface, обрезанный по границам исходного изображения
        clip_rect = original_image.get_rect()
        ideal_source_rect = pygame.Rect(vp_x, vp_y, current_view_width_orig, current_view_height_orig)
        final_source_rect = ideal_source_rect.clip(clip_rect)
        
        # Округляем до целых чисел для pygame.Rect, но только если размеры положительные
        if final_source_rect.width >= 1 and final_source_rect.height >= 1:
            # pygame.Rect ожидает целые числа
            final_source_rect.x = int(round(final_source_rect.x))
            final_source_rect.y = int(round(final_source_rect.y))
            final_source_rect.width = int(round(final_source_rect.width))
            final_source_rect.height = int(round(final_source_rect.height))

            # Дополнительная проверка, т.к. округление могло сделать размеры < 1
            if final_source_rect.width >= 1 and final_source_rect.height >= 1:
                 try:
                    sub_image = original_image.subsurface(final_source_rect)
                    # Используем smoothscale для лучшего качества при масштабировании
                    scaled_sub_image = pygame.transform.scale(sub_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
                    screen.blit(scaled_sub_image, (0, 0))
                 except ValueError as e: # Например, "subsurface rectangle outside surface area"
                    print(f"Этап 1: Ошибка subsurface с rect {final_source_rect} (исходный: {orig_w}x{orig_h}): {e}")
                    # Переходим к Этапу 2 при ошибке, чтобы не застрять
                    animation_stage = 2
                    current_pixel_spacing = initial_pixel_spacing 
            else: # После округления размеры стали невалидными
                print(f"Этап 1: Невалидные размеры для subsurface после округления {final_source_rect}. Переход к Этапу 2.")
                animation_stage = 2
                current_pixel_spacing = initial_pixel_spacing
        else: # Изначально viewport слишком мал или вне границ
            print(f"Этап 1: Viewport {final_source_rect} слишком мал или вне границ. Переход к Этапу 2.")
            animation_stage = 2
            current_pixel_spacing = initial_pixel_spacing

        if save_frames_enabled:
            try:
                pygame.image.save(screen, os.path.join(output_frames_dir, f"frame_{frame_count:05d}.png"))
                frame_count += 1
            except pygame.error as e: print(f"Ошибка сохранения кадра (Этап 1): {e}")

        zoom_scale_stage1 *= STAGE1_ZOOM_SPEED
        
        # Условие перехода к Этапу 2
        # Когда один пиксель оригинала, увеличенный в zoom_scale_stage1 раз,
        # и спроецированный на экран (где всё изображение шириной orig_w/zoom_scale_stage1 занимает SCREEN_WIDTH),
        # станет равен TARGET_PIXEL_SIZE.
        # Размер пикселя оригинала на экране = SCREEN_WIDTH / (orig_w / zoom_scale_stage1)
        if current_view_width_orig > 0: # current_view_width_orig - это ideal orig_w / zoom_scale_stage1
            effective_pixel_width_on_screen = SCREEN_WIDTH / current_view_width_orig
            if effective_pixel_width_on_screen >= TARGET_PIXEL_SIZE:
                print(f"Переход на Этап 2. Эффективная ширина пикселя: {effective_pixel_width_on_screen:.2f} >= Цель: {TARGET_PIXEL_SIZE}")
                animation_stage = 2
                current_pixel_spacing = initial_pixel_spacing # Начинаем Этап 2 с этим расстоянием
        elif zoom_scale_stage1 * (1.0/orig_w if orig_w >0 else 1.0) > SCREEN_WIDTH : # Очень сильно приблизили
             print(f"Переход на Этап 2 из-за высокого zoom_scale_stage1.")
             animation_stage = 2
             current_pixel_spacing = initial_pixel_spacing

    elif animation_stage == 2:
        if not execute_stage2_drawing_logic(): # Эта функция сама сохраняет кадры и обновляет frame_count
            is_animating = False
            animation_stage = 0 # Завершение анимации
    else: # Неизвестный или неактивный этап
        is_animating = False

    if is_animating:
        pygame.display.flip()


def main_loop():
    global is_animating 

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Нажат пробел, сброс анимации.")
                    reset_animation_state() 
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not is_animating:
                    start_animation(event.pos)

        if is_animating:
            process_animation_frame() # Эта функция теперь управляет флагом is_animating

    pygame.quit()

if __name__ == '__main__':
    image_file = parse_arguments()
    init_pygame(image_file)
    main_loop()