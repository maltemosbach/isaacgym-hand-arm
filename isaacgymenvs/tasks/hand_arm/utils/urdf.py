def generate_cuboid_bin_urdf(height, depth_width, file_path):
    urdf_template = f'''
    <?xml version="1.0"?>
    <robot name="cuboid_bin">

      <!-- Materials -->
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>

      <!-- Links -->
      <link name="base_link"/>

      <link name="floor">
            <visual>
            <geometry>
                <box size="{depth_width} {depth_width} 0.01"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} {depth_width} 0.01"/>
            </geometry>
            </collision>
        </link>

      <link name="left_wall">
            <visual>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="right_wall">
            <visual>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="front_wall">
            <visual>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="back_wall">
            <visual>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            </collision>
        </link>


      <!-- Joints -->
      <joint name="floor_joint" type="fixed">
        <parent link="base_link"/>
        <child link="floor"/>
        <origin xyz="0 0 {-height/2}" rpy="0 0 0"/>
      </joint>

      <joint name="left_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="left_wall"/>
            <origin xyz="-{depth_width/2 + 0.005} 0 {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="right_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="right_wall"/>
            <origin xyz="{depth_width/2 + 0.005} 0 {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="front_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="front_wall"/>
            <origin xyz="0 {depth_width/2 + 0.005} {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="back_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="back_wall"/>
            <origin xyz="0 -{depth_width/2 + 0.005} {height/2}" rpy="0 0 0"/>
        </joint>

    </robot>
    '''

    urdf_string = urdf_template.format(height=height, depth_width=depth_width)

    with open(file_path, 'w') as f:
        f.write(urdf_string)


def generate_table_with_hole(x_range, y_range, height, hole_x_range, hole_y_range, file_path):
    urdf_template = f'''
        <?xml version="1.0"?>
        <robot name="table_with_hole">

          <!-- Links -->
          <link name="base_link"/>

          <link name="front_wall">
                <visual>
                <geometry>
                    <box size="{hole_x_range[0] - x_range[0]} {y_range[1] - y_range[0]} {height}"/>
                </geometry>
                </visual>
                <collision>
                <geometry>
                    <box size="{hole_x_range[0] - x_range[0]} {y_range[1] - y_range[0]} {height}"/>
                </geometry>
                </collision>
            </link>

            <link name="back_wall">
                <visual>
                    <geometry>
                        <box size="{x_range[1] - hole_x_range[1]} {y_range[1] - y_range[0]} {height}"/>
                    </geometry>
                </visual>
                <collision>
                    <geometry>
                        <box size="{x_range[1] - hole_x_range[1]} {y_range[1] - y_range[0]} {height}"/>
                    </geometry>
                </collision>
            </link>

            <link name="right_wall">
                <visual>
                <geometry>
                    <box size="{x_range[1] - x_range[0]} {hole_y_range[0] - y_range[0]} {height}"/>
                </geometry>
                </visual>
                <collision>
                    <geometry>
                        <box size="{x_range[1] - x_range[0]} {hole_y_range[0] - y_range[0]} {height}"/>
                    </geometry>
                </collision>
            </link>

            <link name="left_wall">
                <visual>
                <geometry>
                    <box size="{x_range[1] - x_range[0]} {y_range[1] - hole_y_range[1]} {height}"/>
                </geometry>
                </visual>
                <collision>
                    <geometry>
                        <box size="{x_range[1] - x_range[0]} {y_range[1] - hole_y_range[1]} {height}"/>
                    </geometry>
                </collision>
            </link>


          <!-- Joints -->
          <joint name="right_joint" type="fixed">
            <parent link="base_link"/>
            <child link="front_wall"/>
            <origin xyz="{x_range[0] + 0.5 * (hole_x_range[0] - x_range[0])} {y_range[0] + 0.5 * (y_range[1] - y_range[0])} 0" rpy="0 0 0"/>
          </joint>
          
          <joint name="left_joint" type="fixed">
            <parent link="base_link"/>
            <child link="back_wall"/>
            <origin xyz="{(hole_x_range[1] + x_range[1]) / 2} {y_range[0] + 0.5 * (y_range[1] - y_range[0])} 0" rpy="0 0 0"/>
          </joint>
          
          <joint name="front_joint" type="fixed">
            <parent link="base_link"/>
            <child link="right_wall"/>
            <origin xyz="{x_range[0] + 0.5 * (x_range[1] - x_range[0])} {y_range[0] + 0.5 * (hole_y_range[0] - y_range[0])} 0" rpy="0 0 0"/>
          </joint>
          
          <joint name="back_joint" type="fixed">
            <parent link="base_link"/>
            <child link="left_wall"/>
            <origin xyz="{x_range[0] + 0.5 * (x_range[1] - x_range[0])} {(hole_y_range[1] + y_range[1]) / 2} 0" rpy="0 0 0"/>
          </joint>


        </robot>
        '''

    with open(file_path, 'w') as f:
        f.write(urdf_template)
