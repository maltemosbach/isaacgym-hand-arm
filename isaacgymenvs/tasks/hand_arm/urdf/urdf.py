import os
from yourdfpy import URDF
from typing import List, Dict, Tuple, Optional
from lxml import etree
import logging
_logger = logging.getLogger(__name__)
import six
from yourdfpy import Joint

import copy

class HandArmURDF(URDF):
    @staticmethod
    def load(fname_or_file, **kwargs):
        if isinstance(fname_or_file, six.string_types):
            if not os.path.isfile(fname_or_file):
                raise ValueError("{} is not a file".format(fname_or_file))

            if not "mesh_dir" in kwargs:
                kwargs["mesh_dir"] = os.path.dirname(fname_or_file)

        try:
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(fname_or_file, parser=parser)
            xml_root = tree.getroot()
        except Exception as e:
            _logger.error(e)
            _logger.error("Using different parsing approach.")

            events = ("start", "end", "start-ns", "end-ns")
            xml = etree.iterparse(fname_or_file, recover=True, events=events)

            # Iterate through all XML elements
            for action, elem in xml:
                # Skip comments and processing instructions,
                # because they do not have names
                if not (
                    isinstance(elem, etree._Comment)
                    or isinstance(elem, etree._ProcessingInstruction)
                ):
                    # Remove a namespace URI in the element's name
                    # elem.tag = etree.QName(elem).localname
                    if action == "end" and ":" in elem.tag:
                        elem.getparent().remove(elem)

            xml_root = xml.root

        # Remove comments
        etree.strip_tags(xml_root, etree.Comment)
        etree.cleanup_namespaces(xml_root)

        return HandArmURDF(robot=URDF._parse_robot(xml_element=xml_root), **kwargs)

    def attach(self, other, link, prefix="", origin=None, **kwargs):
        #root_urdf = HandArmURDF(
        #    robot=copy.deepcopy(self.robot), build_scene_graph=False, load_meshes=False
        #)

        new_robot = copy.deepcopy(self.robot)

        #for j in other.robot.joints:
        #    root_urdf.robot.joints.append(j)
        #for l in other.robot.links:
        #    root_urdf.robot.links.append(l)

        #root_urdf.robot.joints.append(Joint(
        #    name='attachment_joint',
        #    type='fixed',
        #    parent='wrist_link_3',
        #    child='palm_link',
        #    origin=origin
        #))



        result = HandArmURDF(robot=new_robot, **kwargs)

        print("result == self:", result == self)
        #result = self

        print("result.validate():", result.validate())

        print("result.link_map.keys():", result.link_map.keys())
        print("result.joint_map.keys():", result.joint_map.keys())

        return result

