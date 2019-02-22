def remove_car_from_sun2012():
    from shutil import copyfile
    import xml.etree.cElementTree as ET
    bg_root = '/media/nolan/HDD1/sun2012pascalformat'
    sun_img_path = os.path.join(bg_root, 'JPEGImages')
    sun_anno_path = os.path.join(bg_root, 'Annotations')
    counter = 0
    for img in os.listdir(sun_img_path):
        detected = False
        img_name = (img.split('.')[0]).split('/')[-1]

        img_xml_path = os.path.join(sun_anno_path, (img_name+'.xml'))
        try:
            img_xml = ET.ElementTree(file=img_xml_path)
            root = img_xml.getroot()
            for child in root:
                if child.tag == 'object':
                    for sub_child in child:
                        if sub_child.tag == 'name':
                            text = sub_child.text
                            if ('car' in text or 'van' in text or 'truck' in text):
                                detected = True
                                break
                if detected:
                    break
        except Exception as e:
            pass

        if not detected:
            counter += 1
            src = os.path.join(sun_img_path, img)
            dst = os.path.join('/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/sun2012', img)
            copyfile(src, dst)
    print(counter)
