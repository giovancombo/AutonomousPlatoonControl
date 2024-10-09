from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import NodePath
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from math import pi, sin, cos, radians, sqrt


class PlatooningVisualizer(ShowBase):
    def __init__(self, env):
        ShowBase.__init__(self)
        
        self.env = env
        self.episode = 1
        self.total_reward = 0
        self.avg_reward = 0
        self.instant_rewards = []

        self.car_height_offset = 0
        self.road_width = 8
        self.road_length = 800
        self.road_color = (0.6, 0.6, 0.6, 1)  # Grigio
        self.terrain_width = 10000
        self.terrain_length = 10000
        self.terrain_color = (0, 0.8, 0.35, 1)  # Verde
        self.sidewalk_width = 3
        self.sidewalk_height = 0.25
        self.sidewalk_color = (0.9, 0.9, 0.9, 1)  # Grigio chiaro
        dash_length = 20
        gap_length = 0
        self.leader_starting_pos = -self.road_length/2 + 50
        
        # Disabilita i controlli di camera predefiniti
        self.disableMouse()
        self.camera_control = CameraControl(self)
        
        self.leader = self.loader.loadModel("Models/sedan.glb")
        self.leader.setHpr(180, 0, 0)
        self.leader.setScale(1.55)
        self.leader.reparentTo(self.render)

        self.follower = self.loader.loadModel("Models/police.glb")
        self.follower.setHpr(180, 0, 0)
        self.follower.setScale(1.55)
        self.follower.reparentTo(self.render)
        
        # Crea la strada
        self.road = self.create_plane(self.road_width, self.road_length, self.road_color)
        self.road.setColor(self.road_color)
        self.road.setPos(0, 0, 0)
        self.road.reparentTo(self.render)
        
        # Crea il terreno
        self.terrain = self.create_plane(self.terrain_width, self.terrain_length, self.terrain_color)
        self.terrain.setColor(self.terrain_color)
        self.terrain.setPos(0, 0, -0.1)
        self.terrain.reparentTo(self.render)

        # Crea i marciapiedi
        self.left_sidewalk = self.loader.loadModel("models/box")
        self.left_sidewalk.setScale(self.sidewalk_width, self.road_length, self.sidewalk_height)
        self.left_sidewalk.setPos(-self.road_width/2 - self.sidewalk_width, -self.road_length/2, 0)
        self.left_sidewalk.setColor(self.sidewalk_color)  # Grigio chiaro
        #self.left_sidewalk.reparentTo(self.render)

        self.right_sidewalk = self.loader.loadModel("models/box")
        self.right_sidewalk.setScale(self.sidewalk_width, self.road_length, self.sidewalk_height)
        self.right_sidewalk.setPos(self.road_width/2, -self.road_length/2, 0)
        self.right_sidewalk.setColor(self.sidewalk_color)  # Grigio chiaro
        #self.right_sidewalk.reparentTo(self.render)

        self.building1sx = self.loader.loadModel("Models/city/skyscraperA.glb")
        self.building1sx.setScale(10)
        self.building1sx.setHpr(90, 0, 0)
        self.building1sx.setPos(-self.road_width/2 - 20, 400, 0)
        self.building1sx.reparentTo(self.render)

        self.building2sx = self.loader.loadModel("Models/city/large_buildingA.glb")
        self.building2sx.setScale(10)
        self.building2sx.setHpr(90, 0, 0)
        self.building2sx.setPos(-self.road_width/2 - 20, 350, 0)
        self.building2sx.reparentTo(self.render)

        self.building3sx = self.loader.loadModel("Models/city/low_wideA.glb")
        self.building3sx.setScale(10)
        self.building3sx.setHpr(90, 0, 0)
        self.building3sx.setPos(-self.road_width/2 - 20, 300, 0)
        self.building3sx.reparentTo(self.render)

        self.building4sx = self.loader.loadModel("Models/city/low_buildingA.glb")
        self.building4sx.setScale(10)
        self.building4sx.setHpr(90, 0, 0)
        self.building4sx.setPos(-self.road_width/2 - 20, 250, 0)
        self.building4sx.reparentTo(self.render)

        self.building1dx = self.loader.loadModel("Models/city/large_buildingB.glb")
        self.building1dx.setScale(10)
        self.building1dx.setHpr(-90, 0, 0)
        self.building1dx.setPos(self.road_width/2 + 20, 250, 0)
        self.building1dx.reparentTo(self.render)

        self.building2dx = self.loader.loadModel("Models/city/large_buildingC.glb")
        self.building2dx.setScale(10)
        self.building2dx.setHpr(-90, 0, 0)
        self.building2dx.setPos(self.road_width/2 + 20, 300, 0)
        self.building2dx.reparentTo(self.render)

        self.building3dx = self.loader.loadModel("Models/city/skyscraperE.glb")
        self.building3dx.setScale(10)
        self.building3dx.setHpr(-90, 0, 0)
        self.building3dx.setPos(self.road_width/2 + 20, 350, 0)
        self.building3dx.reparentTo(self.render)

        self.building4dx = self.loader.loadModel("Models/city/low_wideB.glb")
        self.building4dx.setScale(10)
        self.building4dx.setHpr(-90, 0, 0)
        self.building4dx.setPos(self.road_width/2 + 20, 400, 0)
        self.building4dx.reparentTo(self.render)

        self.buildingct = self.loader.loadModel("Models/city/skyscraperD.glb")
        self.buildingct.setScale(10)
        self.buildingct.setHpr(-90, 0, 0)
        self.buildingct.setPos(0, 400, 0)
        self.buildingct.reparentTo(self.render)

        # Crea la linea tratteggiata
        self.center_line = self.create_dashed_line(self.road_length, (1, 1, 1, 1))  # Bianco
        self.center_line.setPos(0, -self.road_length/2, 0.01)
        self.center_line.reparentTo(self.render)

        self.left_line = self.create_dashed_line(self.road_length, (1, 1, 1, 1), dash_length, gap_length)  # Bianco
        self.left_line.setPos(-self.road_width/2 + 0.5, -self.road_length/2, 0.01)
        self.left_line.reparentTo(self.render)

        self.right_line = self.create_dashed_line(self.road_length, (1, 1, 1, 1), dash_length, gap_length)  # Bianco
        self.right_line.setPos(self.road_width/2 - 0.5, -self.road_length/2, 0.01)
        self.right_line.reparentTo(self.render)

        self.desired_distance_line = self.create_line((1, 0, 0, 1))  # Rosso
        self.actual_distance_line = self.create_line((0, 0, 1, 1))   # Blu
        self.desired_distance_line.reparentTo(self.render)
        self.actual_distance_line.reparentTo(self.render)

        self.desired_text = self.create_text("DESIRED", (1, 0, 0, 1))  # Rosso
        self.desired_text.reparentTo(self.render)
        
        # Aggiungi illuminazione
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        self.info_display = self.create_info_display()
        
        # Imposta la telecamera
        self.camera.setPos(0,0,0)
        self.camera.lookAt(0,0,0)
        
        self.is_visualizing = False
        self.paused = False

        self.taskMgr.add(self.update_camera, "UpdateCameraTask")
        
        self.total_distance = 0

    def create_line(self, color):
        line_segs = LineSegs()
        line_segs.setThickness(2)
        line_segs.setColor(color)
        line_segs.moveTo(-1.5, 0, 0.02)  # Inizio della linea
        line_segs.drawTo(1.5, 0, 0.02)   # Fine della linea, lunga 2 unità
        return self.render.attachNewNode(line_segs.create())
    
    def create_text(self, text, color):
        text_node = TextNode('road_text')
        text_node.setText(text)
        text_node.setTextColor(color)
        text_node.setAlign(TextNode.ACenter)
        text_path = self.render.attachNewNode(text_node)
        text_path.setScale(0.5)  # Regola la dimensione del testo
        text_path.setHpr(0, -90, 0)  # Ruota il testo per essere parallelo alla strada
        return text_path

    def create_plane(self, width, length, color):
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('plane', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')  # Rinominato per evitare conflitti
        
        # Define the vertices
        vertex.addData3(-width/2, -length/2, 0)
        vertex.addData3(width/2, -length/2, 0)
        vertex.addData3(width/2, length/2, 0)
        vertex.addData3(-width/2, length/2, 0)
        
        for _ in range(4):
            normal.addData3(0, 0, 1)
            color_writer.addData4(*color)  # Usa il colore passato come argomento
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode('plane')
        node.addGeom(geom)
        
        return NodePath(node)
    
    def create_dashed_line(self, length, color, dash_length=3, gap_length=4, line_width=0.2):
        line_root = NodePath("dashed_line_root")
        
        for i in range(0, int(length), dash_length + gap_length):
            line = self.create_plane(line_width, dash_length, color)
            line.reparentTo(line_root)
            line.setPos(0, i, 0)
        
        return line_root
    
    def create_info_display(self):
        info_text = OnscreenText(
            text="",
            style=1,
            fg=(0, 0, 0, 1),  # Colore nero
            pos=(-1.25, -0.3),  # Posizione in basso a sinistra
            align=TextNode.ALeft,
            scale=0.05,  # Dimensione del testo
            mayChange=True
        )
        return info_text
    
    def reset_episode(self, episode):
        self.episode = episode + 1
        self.total_distance = 0
        self.leader.setPos(1.9, self.leader_starting_pos, self.car_height_offset)
        self.follower.setPos(1.9, self.leader_starting_pos - self.env.actual_distance, self.car_height_offset)
        self.camera_control.reset()
        self.is_visualizing = True
        self.paused = False
        self.instant_rewards = []

    def update_positions(self):
        if self.paused:
            return

        leader_pos = self.total_distance + self.leader_starting_pos
        follower_pos = leader_pos - self.env.actual_distance
        
        self.leader.setPos(1.9, leader_pos, self.car_height_offset)
        self.follower.setPos(1.9, follower_pos, self.car_height_offset)
        
        # Aggiorna il punto medio tra i veicoli
        midpoint = (leader_pos + follower_pos) / 2
        self.camera_control.update_target(Vec3(0, midpoint, self.car_height_offset))

    def update_distance_lines(self):
        leader_pos = self.leader.getPos()
        follower_pos = self.follower.getPos()

        # Aggiorna la linea della desired_distance
        desired_pos = leader_pos - Vec3(0, self.env.desired_distance, 0)
        self.desired_distance_line.setPos(desired_pos.x, desired_pos.y + 2.1, 0.1)
        self.desired_text.setPos(desired_pos.x, desired_pos.y + 2.3, 0.1)
        
        # Aggiorna la linea della actual_distance
        self.actual_distance_line.setPos(follower_pos.x, follower_pos.y + 2.1, 0.1)

    def update_info_display(self, ep, ev, acc, leader_vel, agent_vel, actual_distance, desired_distance):
        info_text = (
            f"Episode: {self.episode}\n"
            f"Position gap (ep): {ep:.2f} m\n"
            f"Velocity gap (ev): {ev:.2f} m/s\n"
            f"Acceleration gap (acc): {acc:.2f} m/s^2\n"
            f"Leader velocity: {(leader_vel * 3.6):.2f} km/h\n"
            f"Agent velocity: {(agent_vel * 3.6):.2f} km/h\n"
            f"Actual distance: {(actual_distance - self.env.vehicles_length):.2f} m\n"
            f"Desired distance: {(desired_distance - self.env.vehicles_length):.2f} m\n\n"
            f"Total Reward: {self.total_reward:.4f}\n"
            f"Avg Reward (last 100): {self.avg_reward:.4f}\n"
            f"Instant Reward: {self.instant_rewards[-1]:.4f}\n\n"
        )
        self.info_display.setText(info_text)

    def update(self, env):
        if not self.is_visualizing:
            self.episode += 1
            return
        
        self.env = env
        self.update_positions()
        self.update_distance_lines()

        ep = self.env.state[0]  # Assumendo che ep sia il primo elemento dello stato
        ev = self.env.state[1]  # Assumendo che ev sia il secondo elemento dello stato
        acc = self.env.state[2]  # Assumendo che acc sia il terzo elemento dello stato
        leader_vel = self.env.leader_velocity  # Assumendo che leader_acc sia un attributo di env
        agent_vel = self.env.agent_velocity
        actual_distance = self.env.actual_distance
        desired_distance = self.env.desired_distance
        self.update_info_display(ep, ev, acc, leader_vel, agent_vel, actual_distance, desired_distance)

    def update_camera(self, task):
        self.camera_control.update()
        return Task.cont

    def toggle_pause(self):
        self.paused = not self.paused

    def stop_visualizing(self):
        self.is_visualizing = False
        self.paused = True


class CameraControl:
    def __init__(self, base):
        self.base = base
        self.camera = base.camera
        
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.8
        self.zoom_speed = 1.5
        self.min_height = 0.5
        
        self.camera_distance = 40
        self.camera_pitch = 20
        self.camera_yaw = 0
        
        self.target = Vec3(0, 0, 0)
        self.offset = Vec3(-40, -40, 70)
        
        # Accetta gli input del mouse
        self.base.accept("mouse3", self.start_drag)
        self.base.accept("mouse3-up", self.stop_drag)
        self.base.accept("wheel_up", self.zoom_in)
        self.base.accept("wheel_down", self.zoom_out)
        
        self.dragging = False

    def start_drag(self):
        self.dragging = True
        self.last_mouse_x = self.base.mouseWatcherNode.getMouseX()
        self.last_mouse_y = self.base.mouseWatcherNode.getMouseY()

    def stop_drag(self):
        self.dragging = False

    def zoom_in(self):
        self.camera_distance = max(5, self.camera_distance - self.zoom_speed)

    def zoom_out(self):
        self.camera_distance = min(100, self.camera_distance + self.zoom_speed)

    def update_target(self, new_target):
        self.target = new_target

    def update(self):
        if self.dragging and self.base.mouseWatcherNode.hasMouse():
            mouse_x = self.base.mouseWatcherNode.getMouseX()
            mouse_y = self.base.mouseWatcherNode.getMouseY()
            
            dx = mouse_x - self.last_mouse_x
            dy = mouse_y - self.last_mouse_y
            
            self.camera_yaw -= dx * self.mouse_sensitivity * 100
            self.camera_pitch += dy * self.mouse_sensitivity * 100
            
            self.camera_pitch = max(-89, min(89, self.camera_pitch))
            
            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y
        
        # Calcola la nuova posizione della camera relativa al target
        relative_x = self.camera_distance * -sin(radians(self.camera_yaw)) * cos(radians(self.camera_pitch))
        relative_y = self.camera_distance * -cos(radians(self.camera_yaw)) * cos(radians(self.camera_pitch))
        relative_z = self.camera_distance * sin(radians(self.camera_pitch))
        
        # Calcola la nuova posizione della camera
        new_pos = self.target + Vec3(relative_x, relative_y, relative_z)
        
        # Controlla se la nuova posizione è sotto il terreno
        if new_pos.z < self.min_height:
            # Calcola l'altezza corretta mantenendo la distanza dal target
            distance_xy = sqrt(relative_x**2 + relative_y**2)
            max_distance_xy = sqrt(self.camera_distance**2 - self.min_height**2)
            if distance_xy > max_distance_xy:
                scale = max_distance_xy / distance_xy
                relative_x *= scale
                relative_y *= scale
            relative_z = self.min_height
            new_pos = self.target + Vec3(relative_x, relative_y, relative_z)
        
        # Aggiorna la posizione e l'orientamento della camera
        self.camera.setPos(new_pos)
        self.camera.lookAt(self.target)

    def reset(self):
        self.camera_distance = 40
        self.camera_pitch = 20
        self.camera_yaw = 0
        self.target = Vec3(0, 0, 0)