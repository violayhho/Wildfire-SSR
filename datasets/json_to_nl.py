def build_clip_caption(feature_dict):
    def get_val(key):
        val = feature_dict.get(key, 'unsure')
        if str(val).lower() in ['unsure', 'unknown', 'none', 'null']:
            return None
        return val

    phrases = []

    # 1. STRUCTURE
    st_type = get_val('structure type')
    if st_type:
        st_type = f"{st_type} structure"
    else:
        st_type = "building"
    phrases.append(f"{st_type}")

    # 2. ROOF (Combined Angle + Material)
    r_mat = get_val('roof material')
    if r_mat:
        r_mat = f"{r_mat} roof"
    else:
        r_mat = "roof"
    r_ang = get_val('roof angle')
    angle_map = {"flat": "flat", "60": "steep", "135": "pitched"}
    r_desc = angle_map.get(str(r_ang), "")  
    phrases.append(f"{r_desc} {r_mat}".strip())

    # 3. SIDING
    siding = get_val('siding material')
    clearance = get_val('siding-to-ground clearance present')
    
    sid_parts = []
    if siding: sid_parts.append(f"{siding} siding")
    if clearance == 'false': sid_parts.append("to ground contact")
    elif clearance == 'true': sid_parts.append("to ground clearance")
    
    if sid_parts: phrases.append(" ".join(sid_parts))

    # 4. DECK & FENCE
    dk_surf = get_val('deck surface material')
    dk_under = get_val('under-deck condition')
    
    if dk_surf or dk_under:
        dk_str = f"{dk_surf or ''} deck".strip()
        if dk_under: dk_str += f" ({dk_under} underneath)"
        phrases.append(dk_str)

    fence = get_val('fence-to-house connection')
    if fence == "no fence": phrases.append("no fence")
    elif fence:
        short_fence = fence.replace("fence attached", "attached fence").replace("fence detached", "detached fence")
        phrases.append(short_fence)

    # 5. TERRAIN & VEG
    slope = get_val('terrain slope')
    ground = get_val('immediate zone ground cover')
    
    terr = []
    if slope == 'flat/gentle': slope = 'flat'
    if slope: terr.append(f"{slope} slope")
    if ground: terr.append(ground)
    if terr: phrases.append(", ".join(terr))

    veg_cond = get_val('overall vegetation condition')
    contact = get_val('vegetation-structure contact')
    ladder = get_val('vertical/ladder fuel continuity')

    if veg_cond: phrases.append(f"{veg_cond} veg")
    
    if contact == 'direct contact': phrases.append("veg touches structure")
    elif contact == 'overhanging canopy': phrases.append("canopy overhangs roof")
    
    if ladder == 'true': phrases.append("ladder fuels")

    # 6. HAZARDS & ACCESS
    p_lines = get_val('power lines')
    if p_lines == 'vegetation near power lines':
        phrases.append("power line encroachment")
    elif p_lines == 'no vegetation near power lines':
        phrases.append("power lines exist")
    
    adj = get_val('adjacent parcel condition')
    if adj == 'overgrown/unmanaged':
        phrases.append("overgrown neighbor")

    road = get_val('road access')
    if road: phrases.append(f"{road} road")

    return ", ".join(phrases)


if __name__ == "__main__":
    sample_feature = {
        "roof material": "tile",
        "roof condition": "clean",
        "roof angle": "135",
        "debris in gutter": "false",
        "siding material": "stucco/cement",
        "siding-to-ground clearance present": "false",
        "building maintenance condition": "good",
        "structure type": "wood",
        "window type": "multi-pane",
        "vent type": "standard screen",
        "eave construction": "enclosed",
        "deck surface material": "non-combustible",
        "under-deck condition": "enclosed",
        "fence-to-house connection": "combustible fence attached",
        "foundation type": "enclosed",
        "immediate zone ground cover": "mowed lawn",
        "vegetation-structure contact": "no contact",
        "vertical/ladder fuel continuity": "false",
        "overall vegetation condition": "green and well-maintained",
        "combustible outbuilding present": "false",
        "firewood pile/fuel tank adjacent to structure": "false",
        "terrain slope": "flat/gentle",
        "adjacent parcel condition": "well-maintained",
        "power lines": "vegetation near power lines",
        "road access": "paved and wide"
    }

    caption = build_clip_caption(sample_feature)
    print(caption)