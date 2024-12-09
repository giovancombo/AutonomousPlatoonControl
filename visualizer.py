from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from math import pi, sin, cos, radians, sqrt

class PlatooningVisualizer(ShowBase):    
    # Costanti di visualizzazione
    ROAD_COLOR = (0.6, 0.6, 0.6, 1)         # Grigio
    TERRAIN_COLOR = (0, 0.8, 0.35, 1)       # Verde
    SIDEWALK_COLOR = (0.9, 0.9, 0.9, 1)     # Grigio chiaro
    LINE_COLOR = (1, 1, 1, 1)               # Bianco
    DESIRED_LINE_COLOR = (1, 0, 0, 1)       # Rosso
    ACTUAL_LINE_COLOR = (0, 0, 1, 1)        # Blu
    
    # Dimensioni dell'ambiente
    ROAD_WIDTH = 8
    ROAD_LENGTH = 800
    TERRAIN_WIDTH = 10000
    TERRAIN_LENGTH = 10000
    SIDEWALK_WIDTH = 3
    SIDEWALK_HEIGHT = 0.25
    
    def __init__(self, env):
        ShowBase.__init__(self)
        self.env = env
        
        # Stato del visualizzatore
        self.episode = 1
        self.total_reward = 0
        self.avg_reward = 0
        self.instant_rewards = []
        self.total_distance = 0
        self.is_visualizing = False
        self.paused = False
        
        # Parametri di visualizzazione
        self.car_height_offset = 0
        self.leader_starting_pos = -self.ROAD_LENGTH/2 + 50
        
        self._setup_camera()
        self._setup_environment()
        self._setup_vehicles()
        self._setup_lighting()
        self._setup_info_display()
        self._setup_event_handlers()
        
    def _setup_camera(self):
        """Inizializza la camera e i suoi controlli"""
        self.disableMouse()
        self.camera_control = CameraControl(self)
        self.camera.setPos(0, 0, 0)
        self.camera.lookAt(0, 0, 0)
        
    def _setup_environment(self):
        """Crea l'ambiente 3D base"""
        self._create_road()
        self._create_terrain()
        self._create_sidewalks()
        self._create_road_lines()
        self._create_distance_indicators()
        
    def _create_road(self):
        """Crea la strada principale"""
        self.road = self._create_plane(self.ROAD_WIDTH, self.ROAD_LENGTH, self.ROAD_COLOR)
        self.road.setColor(self.ROAD_COLOR)
        self.road.setPos(0, 0, 0)
        self.road.reparentTo(self.render)
        
    def _create_terrain(self):
        """Crea il terreno circostante"""
        self.terrain = self._create_plane(self.TERRAIN_WIDTH, self.TERRAIN_LENGTH, self.TERRAIN_COLOR)
        self.terrain.setColor(self.TERRAIN_COLOR)
        self.terrain.setPos(0, 0, -0.1)
        self.terrain.reparentTo(self.render)
        
    def _create_road_lines(self):
        """Crea le linee stradali"""
        # Linea centrale
        self.center_line = self._create_dashed_line(self.ROAD_LENGTH, self.LINE_COLOR)
        self.center_line.setPos(0, -self.ROAD_LENGTH/2, 0.01)
        self.center_line.reparentTo(self.render)
        
        # Linee laterali
        self.left_line = self._create_dashed_line(self.ROAD_LENGTH, self.LINE_COLOR, 20, 0)
        self.left_line.setPos(-self.ROAD_WIDTH/2 + 0.5, -self.ROAD_LENGTH/2, 0.01)
        self.left_line.reparentTo(self.render)
        
        self.right_line = self._create_dashed_line(self.ROAD_LENGTH, self.LINE_COLOR, 20, 0)
        self.right_line.setPos(self.ROAD_WIDTH/2 - 0.5, -self.ROAD_LENGTH/2, 0.01)
        self.right_line.reparentTo(self.render)

    def _create_sidewalks(self):
        """Crea i marciapiedi ai lati della strada"""
        # Marciapiede sinistro
        self.left_sidewalk = self.loader.loadModel("models/box")
        self.left_sidewalk.setScale(self.SIDEWALK_WIDTH, self.ROAD_LENGTH, self.SIDEWALK_HEIGHT)
        self.left_sidewalk.setPos(-self.ROAD_WIDTH/2 - self.SIDEWALK_WIDTH, -self.ROAD_LENGTH/2, 0)
        self.left_sidewalk.setColor(self.SIDEWALK_COLOR)
        self.left_sidewalk.reparentTo(self.render)

        # Marciapiede destro
        self.right_sidewalk = self.loader.loadModel("models/box")
        self.right_sidewalk.setScale(self.SIDEWALK_WIDTH, self.ROAD_LENGTH, self.SIDEWALK_HEIGHT)
        self.right_sidewalk.setPos(self.ROAD_WIDTH/2, -self.ROAD_LENGTH/2, 0)
        self.right_sidewalk.setColor(self.SIDEWALK_COLOR)
        self.right_sidewalk.reparentTo(self.render)
        
    def _setup_vehicles(self):
        """Carica e posiziona i modelli dei veicoli"""
        # Leader
        self.leader = self.loader.loadModel("Panda3D/Models/road/sedan.glb")
        self.leader.setHpr(180, 0, 0)
        self.leader.setScale(1.55)
        self.leader.reparentTo(self.render)
        
        # Follower
        self.follower = self.loader.loadModel("Panda3D/Models/road/police.glb")
        self.follower.setHpr(180, 0, 0)
        self.follower.setScale(1.55)
        self.follower.reparentTo(self.render)
        
    def _setup_lighting(self):
        """Configura l'illuminazione della scena"""
        # Luce ambientale
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Luce direzionale
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
    def _create_distance_indicators(self):
        """Crea gli indicatori di distanza"""
        self.desired_distance_line = self._create_line(self.DESIRED_LINE_COLOR)
        self.actual_distance_line = self._create_line(self.ACTUAL_LINE_COLOR)
        self.desired_distance_line.reparentTo(self.render)
        self.actual_distance_line.reparentTo(self.render)
        
        self.desired_text = self._create_text("DESIRED", self.DESIRED_LINE_COLOR)
        self.desired_text.reparentTo(self.render)
        
    def _setup_info_display(self):
        """Crea il display delle informazioni"""
        self.info_display = OnscreenText(
            text="",
            style=1,
            fg=(0, 0, 0, 1),
            pos=(-1.25, -0.3),
            align=TextNode.ALeft,
            scale=0.05,
            mayChange=True
        )
        
    def _setup_event_handlers(self):
        """Configura gli handler degli eventi"""
        self.taskMgr.add(self.update_camera, "UpdateCameraTask")
        self.accept("p", self.toggle_pause)
        
    def update(self, env):
        """Aggiorna lo stato del visualizzatore"""
        if not self.is_visualizing:
            self.episode += 1
            return
            
        self.env = env
        self._update_positions()
        self._update_distance_lines()
        self._update_info_display()
        
    def _update_positions(self):
        """Aggiorna le posizioni dei veicoli"""
        if self.paused:
            return
            
        leader_pos = self.total_distance + self.leader_starting_pos
        follower_pos = leader_pos - self.env.actual_distance
        
        self.leader.setPos(1.9, leader_pos, self.car_height_offset)
        self.follower.setPos(1.9, follower_pos, self.car_height_offset)
        
        # Aggiorna la camera
        midpoint = (leader_pos + follower_pos) / 2
        self.camera_control.update_target(Vec3(0, midpoint, self.car_height_offset))
        
    def _update_distance_lines(self):
        """Aggiorna gli indicatori di distanza"""
        leader_pos = self.leader.getPos()
        follower_pos = self.follower.getPos()
        
        desired_pos = leader_pos - Vec3(0, self.env.desired_distance, 0)
        self.desired_distance_line.setPos(desired_pos.x, desired_pos.y + 2.1, 0.1)
        self.desired_text.setPos(desired_pos.x, desired_pos.y + 2.3, 0.1)
        self.actual_distance_line.setPos(follower_pos.x, follower_pos.y + 2.1, 0.1)
        
    def _update_info_display(self):
        """Aggiorna il display delle informazioni"""
        ep = self.env.state[0]
        ev = self.env.state[1]
        acc = self.env.state[2]
        info_text = (
            f"Episode: {self.episode}\n"
            f"Position gap (ep): {ep:.2f} m\n"
            f"Velocity gap (ev): {ev:.2f} m/s\n"
            f"Acceleration gap (acc): {acc:.2f} m/s^2\n"
            f"Leader velocity: {(self.env.leader_velocity * 3.6):.2f} km/h\n"
            f"Agent velocity: {(self.env.agent_velocity * 3.6):.2f} km/h\n"
            f"Actual distance: {(self.env.actual_distance - self.env.vehicles_length):.2f} m\n"
            f"Desired distance: {(self.env.desired_distance - self.env.vehicles_length):.2f} m\n\n"
            f"Total Reward: {self.total_reward:.4f}\n"
            f"Avg Reward (last 100): {self.avg_reward:.4f}\n"
            f"Instant Reward: {self.instant_rewards[-1]:.4f}\n\n"
        )
        self.info_display.setText(info_text)
        
    def reset_episode(self, episode):
        """Resetta lo stato per un nuovo episodio"""
        self.episode = episode + 1
        self.total_distance = 0
        self.leader.setPos(1.9, self.leader_starting_pos, self.car_height_offset)
        self.follower.setPos(1.9, self.leader_starting_pos - self.env.actual_distance, self.car_height_offset)
        self.camera_control.reset()
        self.is_visualizing = True
        self.paused = False
        self.instant_rewards = []
        
    def update_camera(self, task):
        """Task di aggiornamento della camera"""
        self.camera_control.update()
        return Task.cont
        
    def toggle_pause(self):
        """Toggle dello stato di pausa"""
        self.paused = not self.paused
        
    def stop_visualizing(self):
        """Ferma la visualizzazione"""
        self.is_visualizing = False
        self.paused = True
        
    # Metodi di utility per la creazione di oggetti grafici
    def _create_plane(self, width, length, color):
        """Crea un piano geometrico"""
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('plane', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        vertex.addData3(-width/2, -length/2, 0)
        vertex.addData3(width/2, -length/2, 0)
        vertex.addData3(width/2, length/2, 0)
        vertex.addData3(-width/2, length/2, 0)
        
        for _ in range(4):
            normal.addData3(0, 0, 1)
            color_writer.addData4(*color)
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode('plane')
        node.addGeom(geom)
        
        return NodePath(node)
        
    def _create_line(self, color):
        """Crea una linea"""
        line_segs = LineSegs()
        line_segs.setThickness(2)
        line_segs.setColor(color)
        line_segs.moveTo(-1.5, 0, 0.02)
        line_segs.drawTo(1.5, 0, 0.02)
        return self.render.attachNewNode(line_segs.create())
        
    def _create_text(self, text, color):
        """Crea un nodo di testo"""
        text_node = TextNode('road_text')
        text_node.setText(text)
        text_node.setTextColor(color)
        text_node.setAlign(TextNode.ACenter)
        text_path = self.render.attachNewNode(text_node)
        text_path.setScale(0.5)
        text_path.setHpr(0, -90, 0)
        return text_path
        
    def _create_dashed_line(self, length, color, dash_length=3, gap_length=4, line_width=0.2):
        """Crea una linea tratteggiata"""
        line_root = NodePath("dashed_line_root")
        
        for i in range(0, int(length), dash_length + gap_length):
            line = self._create_plane(line_width, dash_length, color)
            line.reparentTo(line_root)
            line.setPos(0, i, 0)
        
        return line_root

class CameraControl:
    """Gestisce il controllo della camera e l'input dell'utente"""
    
    def __init__(self, base):
        self.base = base
        self.camera = base.camera
        
        # Parametri del mouse
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.8
        self.dragging = False
        
        # Parametri della camera
        self.zoom_speed = 1.5
        self.min_height = 0.5
        self.camera_distance = 40
        self.camera_pitch = 20
        self.camera_yaw = 0
        
        # Target e offset della camera
        self.target = Vec3(0, 0, 0)
        self.offset = Vec3(-40, -40, 70)
        
        # Setup degli input handlers
        self._setup_input_handlers()
        
    def _setup_input_handlers(self):
        """Configura gli handler degli input del mouse"""
        self.base.accept("mouse3", self.start_drag)
        self.base.accept("mouse3-up", self.stop_drag)
        self.base.accept("wheel_up", self.zoom_in)
        self.base.accept("wheel_down", self.zoom_out)

    def start_drag(self):
        """Inizia il dragging della camera"""
        self.dragging = True
        self.last_mouse_x = self.base.mouseWatcherNode.getMouseX()
        self.last_mouse_y = self.base.mouseWatcherNode.getMouseY()

    def stop_drag(self):
        """Termina il dragging della camera"""
        self.dragging = False

    def zoom_in(self):
        """Zoom in della camera"""
        self.camera_distance = max(5, self.camera_distance - self.zoom_speed)

    def zoom_out(self):
        """Zoom out della camera"""
        self.camera_distance = min(100, self.camera_distance + self.zoom_speed)

    def update_target(self, new_target):
        """Aggiorna il punto target della camera"""
        self.target = new_target

    def update(self):
        """Aggiorna la posizione e rotazione della camera"""
        # Gestisce il dragging del mouse
        if self.dragging and self.base.mouseWatcherNode.hasMouse():
            mouse_x = self.base.mouseWatcherNode.getMouseX()
            mouse_y = self.base.mouseWatcherNode.getMouseY()
            
            dx = mouse_x - self.last_mouse_x
            dy = mouse_y - self.last_mouse_y
            
            self.camera_yaw -= dx * self.mouse_sensitivity * 100
            self.camera_pitch += dy * self.mouse_sensitivity * 100
            
            # Limita il pitch per evitare inversioni della camera
            self.camera_pitch = max(-89, min(89, self.camera_pitch))
            
            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y
        
        # Calcola la nuova posizione della camera
        relative_x = self.camera_distance * -sin(radians(self.camera_yaw)) * cos(radians(self.camera_pitch))
        relative_y = self.camera_distance * -cos(radians(self.camera_yaw)) * cos(radians(self.camera_pitch))
        relative_z = self.camera_distance * sin(radians(self.camera_pitch))
        
        new_pos = self.target + Vec3(relative_x, relative_y, relative_z)
        
        # Gestisce il limite di altezza minima
        if new_pos.z < self.min_height:
            distance_xy = sqrt(relative_x**2 + relative_y**2)
            max_distance_xy = sqrt(self.camera_distance**2 - self.min_height**2)
            if distance_xy > max_distance_xy:
                scale = max_distance_xy / distance_xy
                relative_x *= scale
                relative_y *= scale
            relative_z = self.min_height
            new_pos = self.target + Vec3(relative_x, relative_y, relative_z)
        
        # Aggiorna la posizione e orientamento della camera
        self.camera.setPos(new_pos)
        self.camera.lookAt(self.target)

    def reset(self):
        """Resetta la camera alla posizione iniziale"""
        self.camera_distance = 40
        self.camera_pitch = 20
        self.camera_yaw = 0
        self.target = Vec3(0, 0, 0)