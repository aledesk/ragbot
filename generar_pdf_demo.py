"""
Genera un PDF de demostración de lista de precios mayorista.
Ejecutar una sola vez para crear los datos de prueba.
"""

import fitz  # PyMuPDF
import os

def crear_pdf_mayorista():
    doc = fitz.open()

    # ─── Página 1: Portada + Condiciones ────────────────────────────────────────
    page = doc.new_page(width=595, height=842)

    page.draw_rect(fitz.Rect(0, 0, 595, 120), color=None, fill=(0.05, 0.10, 0.20))
    page.insert_text((40, 55), "DISTRIBUIDORA NORTE", fontsize=26, color=(1,1,1), fontname="helv")
    page.insert_text((40, 85), "Lista de Precios Mayoristas — Vigencia: Junio 2025", fontsize=11, color=(0.7,0.8,1.0), fontname="helv")

    y = 150
    page.insert_text((40, y), "CONDICIONES COMERCIALES", fontsize=14, color=(0.05,0.10,0.20), fontname="helv")
    y += 30
    condiciones = [
        "• Precios expresados en USD. Facturación en pesos al tipo de cambio del día.",
        "• Pago al contado: 5% de descuento sobre el total de la factura.",
        "• Pago a 30 días: precio de lista sin descuento.",
        "• Pago a 60 días: recargo del 8% sobre el total.",
        "• Pedido mínimo: $500 USD por orden.",
        "• Flete incluido para pedidos superiores a $1.500 USD (zona AMBA).",
        "• Flete a cargo del comprador para zonas del interior del país.",
        "• Devoluciones: se aceptan hasta 15 días desde la fecha de recepción.",
        "  El producto debe estar sin uso y en su embalaje original.",
        "• Stock sujeto a disponibilidad al momento de confirmación del pedido.",
        "• Precios actualizados el primer lunes de cada mes.",
    ]
    for c in condiciones:
        page.insert_text((40, y), c, fontsize=10, color=(0.2,0.2,0.2))
        y += 20

    y += 20
    page.insert_text((40, y), "ZONAS DE ENTREGA", fontsize=14, color=(0.05,0.10,0.20), fontname="helv")
    y += 30
    zonas = [
        "• AMBA (GBA Norte, Sur, Oeste y CABA): entrega en 48 hs hábiles.",
        "• Rosario, Córdoba, Mendoza: entrega en 5 días hábiles.",
        "• Resto del interior: entrega en 7-10 días hábiles vía correo.",
        "• Retiro en depósito: Av. Constituyentes 4500, CABA — L a V de 9 a 17 hs.",
    ]
    for z in zonas:
        page.insert_text((40, y), z, fontsize=10, color=(0.2,0.2,0.2))
        y += 20

    # ─── Página 2: Categoría Electrónica ────────────────────────────────────────
    page2 = doc.new_page(width=595, height=842)
    page2.draw_rect(fitz.Rect(0, 0, 595, 80), color=None, fill=(0.05, 0.10, 0.20))
    page2.insert_text((40, 50), "ELECTRÓNICA Y ACCESORIOS", fontsize=18, color=(1,1,1), fontname="helv")

    headers = ["SKU", "Descripción", "Marca", "Stock", "Precio USD"]
    productos_elec = [
        ("EL-001", "Auriculares Bluetooth Over-Ear ANC", "SoundMax", "85 u.", "$42.00"),
        ("EL-002", "Auriculares In-Ear True Wireless", "SoundMax", "120 u.", "$28.50"),
        ("EL-003", "Parlante Bluetooth 20W resistente al agua", "BoomBox", "60 u.", "$55.00"),
        ("EL-004", "Parlante Bluetooth 5W portátil", "BoomBox", "95 u.", "$22.00"),
        ("EL-005", "Cargador USB-C 65W GaN", "PowerFast", "200 u.", "$18.00"),
        ("EL-006", "Cargador USB-C 30W compacto", "PowerFast", "310 u.", "$11.50"),
        ("EL-007", "Cable USB-C a USB-C 1m 100W", "CableMax", "500 u.", "$4.20"),
        ("EL-008", "Cable USB-C a Lightning 1m MFi", "CableMax", "180 u.", "$6.80"),
        ("EL-009", "Power Bank 20.000 mAh USB-C+A", "PowerFast", "75 u.", "$35.00"),
        ("EL-010", "Power Bank 10.000 mAh slim", "PowerFast", "130 u.", "$22.00"),
        ("EL-011", "Hub USB-C 7 en 1 (HDMI 4K, USB 3.0 x3)", "TechHub", "50 u.", "$29.00"),
        ("EL-012", "Teclado inalámbrico slim Bluetooth", "TypePro", "40 u.", "$24.00"),
        ("EL-013", "Mouse inalámbrico silencioso 1600 DPI", "TypePro", "90 u.", "$14.00"),
        ("EL-014", "Webcam 1080p con micrófono", "VisioCam", "35 u.", "$38.00"),
        ("EL-015", "Soporte de aluminio para notebook", "DeskPro", "55 u.", "$19.00"),
    ]

    y = 110
    col_x = [30, 95, 300, 390, 460, 535]
    page2.draw_rect(fitz.Rect(25, y-5, 570, y+20), color=None, fill=(0.85,0.90,0.95))
    for i, h in enumerate(headers):
        page2.insert_text((col_x[i], y+12), h, fontsize=9, color=(0.05,0.10,0.20), fontname="helv")
    y += 30

    for idx, (sku, desc, marca, stock, precio) in enumerate(productos_elec):
        if idx % 2 == 0:
            page2.draw_rect(fitz.Rect(25, y-5, 570, y+16), color=None, fill=(0.97,0.97,0.99))
        row = [sku, desc, marca, stock, precio]
        for i, val in enumerate(row):
            page2.insert_text((col_x[i], y+8), val, fontsize=8.5, color=(0.15,0.15,0.15))
        y += 22

    # ─── Página 3: Hogar y Cocina ────────────────────────────────────────────────
    page3 = doc.new_page(width=595, height=842)
    page3.draw_rect(fitz.Rect(0, 0, 595, 80), color=None, fill=(0.10, 0.20, 0.10))
    page3.insert_text((40, 50), "HOGAR Y COCINA", fontsize=18, color=(1,1,1), fontname="helv")

    productos_hogar = [
        ("HG-001", "Pava eléctrica 1.7L acero inox.", "HomeChef", "70 u.", "$31.00"),
        ("HG-002", "Licuadora 600W 1.5L vaso vidrio", "HomeChef", "45 u.", "$48.00"),
        ("HG-003", "Tostadora 4 ranuras 1500W", "HomeChef", "38 u.", "$27.00"),
        ("HG-004", "Sandwichera/waflera desmontable", "HomeChef", "60 u.", "$22.00"),
        ("HG-005", "Freidora de aire 5.5L digital", "AirFry Pro", "25 u.", "$89.00"),
        ("HG-006", "Cafetera espresso 15 bar", "CaféMax", "20 u.", "$112.00"),
        ("HG-007", "Cafetera de filtro 12 tazas", "CaféMax", "55 u.", "$34.00"),
        ("HG-008", "Juego de sartenes antiadherentes x3", "CookPro", "40 u.", "$45.00"),
        ("HG-009", "Set cuchillos acero inox x5 + bloque", "CookPro", "30 u.", "$38.00"),
        ("HG-010", "Organizador modular cajones cocina x4", "SpaceMax", "80 u.", "$16.00"),
        ("HG-011", "Aspiradora sin cable 22V 2 baterías", "CleanPro", "18 u.", "$95.00"),
        ("HG-012", "Mopa eléctrica plana recargable", "CleanPro", "28 u.", "$42.00"),
        ("HG-013", "Purificador de aire HEPA 30m²", "AirFresh", "15 u.", "$78.00"),
        ("HG-014", "Humidificador ultrasónico 4L", "AirFresh", "35 u.", "$29.00"),
        ("HG-015", "Balanza digital cocina 5kg", "HomeChef", "100 u.", "$9.50"),
    ]

    y = 110
    page3.draw_rect(fitz.Rect(25, y-5, 570, y+20), color=None, fill=(0.85,0.95,0.87))
    for i, h in enumerate(headers):
        page3.insert_text((col_x[i], y+12), h, fontsize=9, color=(0.05,0.20,0.05), fontname="helv")
    y += 30

    for idx, (sku, desc, marca, stock, precio) in enumerate(productos_hogar):
        if idx % 2 == 0:
            page3.draw_rect(fitz.Rect(25, y-5, 570, y+16), color=None, fill=(0.97,0.99,0.97))
        row = [sku, desc, marca, stock, precio]
        for i, val in enumerate(row):
            page3.insert_text((col_x[i], y+8), val, fontsize=8.5, color=(0.15,0.15,0.15))
        y += 22

    # ─── Página 4: Descuentos por volumen ────────────────────────────────────────
    page4 = doc.new_page(width=595, height=842)
    page4.draw_rect(fitz.Rect(0, 0, 595, 80), color=None, fill=(0.20, 0.08, 0.02))
    page4.insert_text((40, 50), "DESCUENTOS POR VOLUMEN Y CONTACTO", fontsize=16, color=(1,1,1), fontname="helv")

    y = 110
    page4.insert_text((40, y), "ESCALA DE DESCUENTOS POR MONTO DE PEDIDO", fontsize=13, color=(0.20,0.08,0.02), fontname="helv")
    y += 30

    escalas = [
        ("$500 - $999 USD",   "Sin descuento adicional", "Precio de lista"),
        ("$1.000 - $2.499 USD","3% sobre total",          "Aplica automático"),
        ("$2.500 - $4.999 USD","6% sobre total",          "Aplica automático"),
        ("$5.000 - $9.999 USD","10% sobre total",         "Requiere aprobación comercial"),
        ("$10.000 USD o más",  "15% + condiciones especiales", "Negociación directa con ventas"),
    ]

    desc_headers = ["Rango de compra", "Descuento", "Observación"]
    page4.draw_rect(fitz.Rect(25, y-5, 570, y+20), color=None, fill=(0.95,0.88,0.80))
    dx = [30, 200, 360]
    for i, h in enumerate(desc_headers):
        page4.insert_text((dx[i], y+12), h, fontsize=9, color=(0.20,0.08,0.02), fontname="helv")
    y += 30

    for idx, row in enumerate(escalas):
        if idx % 2 == 0:
            page4.draw_rect(fitz.Rect(25, y-5, 570, y+16), color=None, fill=(0.99,0.97,0.95))
        for i, val in enumerate(row):
            page4.insert_text((dx[i], y+8), val, fontsize=8.5, color=(0.15,0.15,0.15))
        y += 22

    y += 30
    page4.insert_text((40, y), "CONTACTO Y CANALES DE VENTA", fontsize=13, color=(0.20,0.08,0.02), fontname="helv")
    y += 30
    contacto = [
        "Ejecutivo de ventas zona Norte: Carlos Méndez — carlos.mendez@distribuidoranorte.com",
        "Ejecutivo de ventas zona Sur/Centro: Valeria Ríos — v.rios@distribuidoranorte.com",
        "Atención al cliente: ventas@distribuidoranorte.com",
        "WhatsApp Business: +54 9 11 4455-6677 (L a V 9-18 hs)",
        "Portal de pedidos online: pedidos.distribuidoranorte.com",
        "Central telefónica: (011) 4500-1234 int. 2 (Ventas)",
    ]
    for c in contacto:
        page4.insert_text((40, y), c, fontsize=10, color=(0.2,0.2,0.2))
        y += 22

    output_path = "/home/claude/rag-mayorista/data/lista_precios_mayorista.pdf"
    doc.save(output_path)
    doc.close()
    print(f"✅ PDF generado: {output_path}")
    return output_path

if __name__ == "__main__":
    crear_pdf_mayorista()
