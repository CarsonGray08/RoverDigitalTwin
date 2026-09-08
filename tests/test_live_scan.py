import math
import queue
import socket
import struct
import time
import unittest
import uuid

from digital_twin.live_scan import (
    FLAG_HAS_NORMALS,
    FLAG_TEXTURED,
    MSG_CONTROL_DISABLE_IMU,
    MSG_CONTROL_ENABLE_IMU,
    MSG_FINALIZE,
    MSG_IMU,
    MSG_KEYFRAME_IMAGE,
    MSG_KEYFRAME_RGBD,
    MSG_PHONE_POSE,
    TRACKING_LIMITED,
    TRACKING_NORMAL,
    LiveMeshAnchorFrame,
    LiveScanProtocolError,
    LiveScanReceiver,
    LiveScanState,
    PhonePoseStore,
    decode_payload,
    encode_control,
    encode_finalize,
    encode_frame,
    encode_imu,
    encode_keyframe_image,
    encode_keyframe_rgbd,
    encode_phone_pose,
    identity_matrix4,
    rasterize_xy_points,
    transform_normals_to_twin,
    transform_vertices_to_twin,
)


class LiveScanProtocolTests(unittest.TestCase):
    def test_frame_round_trip(self):
        anchor_id = str(uuid.uuid4())
        frame = LiveMeshAnchorFrame(
            anchor_id=anchor_id,
            timestamp_s=12.5,
            anchor_transform=identity_matrix4(),
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            faces=((0, 1, 2),),
            colors=((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        )

        encoded = encode_frame(frame)
        payload_length = struct.unpack_from("<I", encoded)[0]
        decoded = decode_payload(encoded[4:])

        self.assertEqual(payload_length, len(encoded) - 4)
        self.assertEqual(decoded.anchor_id, anchor_id)
        self.assertEqual(decoded.vertices, frame.vertices)
        self.assertEqual(decoded.faces, frame.faces)
        self.assertEqual(decoded.colors, frame.colors)

    def test_frame_without_normals_sets_no_flag(self):
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=1.0,
            vertices=((0.0, 0.0, 0.0),),
            faces=((0, 0, 0),),
            colors=((1, 2, 3),),
        )
        encoded = encode_frame(frame)
        flags = struct.unpack_from("<H", encoded, 4 + 6)[0]
        self.assertEqual(flags & FLAG_HAS_NORMALS, 0)
        self.assertEqual(decode_payload(encoded[4:]).normals, ())

    def test_normals_round_trip_and_flag(self):
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=3.0,
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            faces=((0, 1, 0),),
            colors=((10, 20, 30), (40, 50, 60)),
            normals=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        encoded = encode_frame(frame)
        flags = struct.unpack_from("<H", encoded, 4 + 6)[0]
        self.assertEqual(flags & FLAG_HAS_NORMALS, FLAG_HAS_NORMALS)
        decoded = decode_payload(encoded[4:])
        self.assertEqual(decoded.normals, frame.normals)

    def test_rejects_normals_length_mismatch(self):
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=1.0,
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            faces=((0, 1, 0),),
            colors=((1, 1, 1), (2, 2, 2)),
            normals=((0.0, 0.0, 1.0),),
        )
        with self.assertRaises(LiveScanProtocolError):
            encode_frame(frame)

    def test_keyframe_image_round_trip(self):
        image_id = str(uuid.uuid4())
        jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
        encoded = encode_keyframe_image(image_id, 960, 720, jpeg, timestamp_s=4.0)
        decoded = decode_payload(encoded[4:])
        self.assertEqual(decoded.message_type, MSG_KEYFRAME_IMAGE)
        self.assertTrue(decoded.is_keyframe_image)
        self.assertEqual(decoded.anchor_id, image_id)
        self.assertEqual(decoded.image_width, 960)
        self.assertEqual(decoded.image_height, 720)
        self.assertEqual(decoded.image_bytes, jpeg)

    def test_keyframe_image_pose_and_intrinsics_round_trip(self):
        image_id = str(uuid.uuid4())
        jpeg = b"\xff\xd8jpeg\xff\xd9"
        pose = tuple(float(i) * 0.5 for i in range(16))
        encoded = encode_keyframe_image(
            image_id, 640, 480, jpeg, timestamp_s=2.0,
            camera_transform=pose, intrinsics=(500.0, 501.0, 320.0, 240.0),
        )
        decoded = decode_payload(encoded[4:])
        self.assertEqual(decoded.anchor_transform, pose)
        for got, want in zip(decoded.intrinsics, (500.0, 501.0, 320.0, 240.0)):
            self.assertAlmostEqual(got, want, places=4)

    def test_keyframe_image_stores_pose_in_state(self):
        image_id = str(uuid.uuid4())
        pose = tuple(float(i) for i in range(16))
        frame = decode_payload(
            encode_keyframe_image(
                image_id, 320, 240, b"jpg", camera_transform=pose,
                intrinsics=(1.0, 2.0, 3.0, 4.0),
            )[4:]
        )
        state = LiveScanState()
        state.apply_image(frame)
        kf = state.keyframes[image_id]
        self.assertEqual(kf.width, 320)
        self.assertEqual(kf.height, 240)
        self.assertEqual(kf.camera_transform, pose)
        self.assertEqual(kf.intrinsics, (1.0, 2.0, 3.0, 4.0))

    def test_keyframe_rgbd_round_trip(self):
        image_id = str(uuid.uuid4())
        depth = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        confidence = bytes((0, 1, 2, 2))
        encoded = encode_keyframe_rgbd(
            image_id,
            640,
            480,
            b"\xff\xd8jpeg\xff\xd9",
            2,
            2,
            depth,
            confidence,
            timestamp_s=9.0,
            intrinsics=(10.0, 11.0, 12.0, 13.0),
        )
        decoded = decode_payload(encoded[4:])
        self.assertEqual(decoded.message_type, MSG_KEYFRAME_RGBD)
        self.assertTrue(decoded.is_keyframe_image)
        self.assertEqual(decoded.anchor_id, image_id)
        self.assertEqual(decoded.depth_width, 2)
        self.assertEqual(decoded.depth_height, 2)
        self.assertEqual(decoded.depth_bytes, depth)
        self.assertEqual(decoded.confidence_bytes, confidence)
        self.assertEqual(decoded.intrinsics, (10.0, 11.0, 12.0, 13.0))

    def test_keyframe_rgbd_stores_depth_in_state(self):
        image_id = str(uuid.uuid4())
        depth = struct.pack("<f", 1.25)
        frame = decode_payload(
            encode_keyframe_rgbd(
                image_id,
                320,
                240,
                b"jpg",
                1,
                1,
                depth,
                b"\x02",
            )[4:]
        )
        state = LiveScanState()
        state.apply_image(frame)
        kf = state.keyframes[image_id]
        self.assertEqual(kf.depth_width, 1)
        self.assertEqual(kf.depth_height, 1)
        self.assertEqual(kf.depth_bytes, depth)
        self.assertEqual(kf.confidence_bytes, b"\x02")

    def test_rejects_keyframe_rgbd_depth_length_mismatch(self):
        with self.assertRaises(LiveScanProtocolError):
            encode_keyframe_rgbd(
                str(uuid.uuid4()),
                320,
                240,
                b"jpg",
                2,
                2,
                b"\x00" * 4,
            )

    def test_finalize_round_trip(self):
        decoded = decode_payload(encode_finalize(7.0)[4:])
        self.assertEqual(decoded.message_type, MSG_FINALIZE)
        self.assertTrue(decoded.is_finalize)

    def test_imu_round_trip(self):
        decoded = decode_payload(encode_imu((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), timestamp_s=9.0)[4:])
        self.assertEqual(decoded.message_type, MSG_IMU)
        self.assertTrue(decoded.is_imu)
        self.assertEqual(decoded.imu_accel, (1.0, 2.0, 3.0))
        for got, want in zip(decoded.imu_gyro, (0.1, 0.2, 0.3)):
            self.assertAlmostEqual(got, want, places=6)

    def test_imu_control_round_trip(self):
        for message_type in (MSG_CONTROL_ENABLE_IMU, MSG_CONTROL_DISABLE_IMU):
            decoded = decode_payload(encode_control(message_type, timestamp_s=3.0)[4:])
            self.assertEqual(decoded.message_type, message_type)
            self.assertAlmostEqual(decoded.timestamp_s, 3.0)

    def test_textured_anchor_round_trip(self):
        image_id = str(uuid.uuid4())
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=5.0,
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            faces=((0, 1, 0),),
            colors=((10, 20, 30), (40, 50, 60)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            texture_image_id=image_id,
            uvs=((0.1, 0.2), (0.3, 0.4)),
        )
        encoded = encode_frame(frame)
        flags = struct.unpack_from("<H", encoded, 4 + 6)[0]
        self.assertEqual(flags & FLAG_TEXTURED, FLAG_TEXTURED)
        self.assertEqual(flags & FLAG_HAS_NORMALS, FLAG_HAS_NORMALS)
        decoded = decode_payload(encoded[4:])
        self.assertEqual(decoded.texture_image_id, image_id)
        self.assertEqual(len(decoded.uvs), len(frame.uvs))
        for got, want in zip(decoded.uvs, frame.uvs):
            self.assertAlmostEqual(got[0], want[0], places=5)
            self.assertAlmostEqual(got[1], want[1], places=5)
        self.assertEqual(decoded.normals, frame.normals)

    def test_rejects_uv_length_mismatch(self):
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=1.0,
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            faces=((0, 1, 0),),
            colors=((1, 1, 1), (2, 2, 2)),
            texture_image_id=str(uuid.uuid4()),
            uvs=((0.0, 0.0),),
        )
        with self.assertRaises(LiveScanProtocolError):
            encode_frame(frame)

    def test_rejects_out_of_range_face_index(self):
        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=1.0,
            anchor_transform=identity_matrix4(),
            vertices=((0.0, 0.0, 0.0),),
            faces=((0, 1, 0),),
            colors=((255, 255, 255),),
        )

        with self.assertRaises(LiveScanProtocolError):
            decode_payload(encode_frame(frame)[4:])


class LiveScanTransformTests(unittest.TestCase):
    def test_converts_arkit_y_up_to_usd_z_up(self):
        self.assertEqual(
            transform_vertices_to_twin(((1.0, 2.0, 3.0),), identity_matrix4()),
            ((1.0, -3.0, 2.0),),
        )

    def test_applies_non_identity_anchor_transform(self):
        # Locks the anchor-transform convention the iOS encoder must produce:
        # row-major, row-vector (translation in indices 12-14). The matrix is
        # asymmetric (a shear in row 1) plus a translation so it catches both a
        # transposed rotation and a dropped translation. Row-vector transform of
        # local (1, 1, 1):
        #   x' = 1*1 + 1*2 + 1*0 + 5 = 8
        #   y' = 1*0 + 1*1 + 1*0 + 6 = 7
        #   z' = 1*0 + 1*0 + 1*1 + 7 = 8
        # then the fixed ARKit->USD axis map (x, y, z) -> (x, -z, y): (8, -8, 7).
        anchor_transform = (
            1.0, 0.0, 0.0, 0.0,
            2.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        )

        self.assertEqual(
            transform_vertices_to_twin(((1.0, 1.0, 1.0),), anchor_transform),
            ((8.0, -8.0, 7.0),),
        )

    def test_normals_rotate_with_axis_swap_and_ignore_translation(self):
        # A registration with both a rotation-free translation and the fixed
        # ARKit->USD axis swap (x, y, z) -> (x, -z, y). An up normal (0, 1, 0)
        # must become (0, 0, 1) and must NOT pick up the translation.
        registration = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            10.0, 20.0, 30.0, 1.0,
        )
        self.assertEqual(
            transform_normals_to_twin(((0.0, 1.0, 0.0),), identity_matrix4(), registration),
            ((0.0, 0.0, 1.0),),
        )

    def test_normals_are_renormalized(self):
        result = transform_normals_to_twin(((0.0, 5.0, 0.0),), identity_matrix4())
        self.assertEqual(result, ((0.0, 0.0, 1.0),))

    def test_applies_registration_translation_after_axis_conversion(self):
        registration = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            10.0,
            20.0,
            30.0,
            1.0,
        )

        self.assertEqual(
            transform_vertices_to_twin(((1.0, 2.0, 3.0),), identity_matrix4(), registration),
            ((11.0, 17.0, 32.0),),
        )


class LiveScanStateTests(unittest.TestCase):
    def test_anchor_upsert_replaces_previous_geometry(self):
        anchor_id = str(uuid.uuid4())
        state = LiveScanState()
        state.apply_frame(
            LiveMeshAnchorFrame(
                anchor_id=anchor_id,
                timestamp_s=1.0,
                vertices=((0.0, 0.2, -1.0),),
                colors=((255, 255, 255),),
            ),
            now_s=10.0,
        )
        state.apply_frame(
            LiveMeshAnchorFrame(
                anchor_id=anchor_id,
                timestamp_s=2.0,
                vertices=((2.0, 0.2, -1.0),),
                colors=((255, 255, 255),),
            ),
            now_s=11.0,
        )

        self.assertEqual(len(state.anchors), 1)
        self.assertEqual(state.known_points_xy(), ((2.0, 1.0),))

    def test_obstacle_band_and_decay(self):
        anchor_id = str(uuid.uuid4())
        state = LiveScanState()
        state.apply_frame(
            LiveMeshAnchorFrame(
                anchor_id=anchor_id,
                timestamp_s=1.0,
                vertices=(
                    (0.0, 0.2, -1.0),  # USD z 0.2, in obstacle band
                    (1.0, 1.5, -1.0),  # USD z 1.5, above obstacle band
                ),
                colors=((255, 255, 255), (255, 255, 255)),
            ),
            now_s=10.0,
        )

        self.assertEqual(state.obstacle_points_xy((0.12, 0.45), now_s=10.5, decay_s=2.0), ((0.0, 1.0),))
        self.assertEqual(state.obstacle_points_xy((0.12, 0.45), now_s=13.0, decay_s=2.0), ())
        self.assertEqual(state.anchors, {})

    def test_stores_image_and_tracks_texture_references(self):
        image_id = str(uuid.uuid4())
        anchor_id = str(uuid.uuid4())
        state = LiveScanState()
        state.apply_image(
            LiveMeshAnchorFrame(
                anchor_id=image_id,
                timestamp_s=1.0,
                image_width=4,
                image_height=4,
                image_bytes=b"jpegbytes",
                message_type=MSG_KEYFRAME_IMAGE,
            )
        )
        self.assertEqual(state.images[image_id], b"jpegbytes")
        self.assertEqual(state.referenced_image_ids(), set())

        state.apply_frame(
            LiveMeshAnchorFrame(
                anchor_id=anchor_id,
                timestamp_s=2.0,
                vertices=((0.0, 0.0, 0.0),),
                colors=((1, 2, 3),),
                texture_image_id=image_id,
                uvs=((0.5, 0.5),),
            ),
            now_s=10.0,
        )
        self.assertEqual(state.referenced_image_ids(), {image_id})
        anchor = state.anchors[anchor_id]
        self.assertEqual(anchor.texture_image_id, image_id)
        self.assertEqual(anchor.uvs, ((0.5, 0.5),))

    def test_rasterizes_negative_coordinates_with_floor_cells(self):
        grid = rasterize_xy_points(((-0.1, -0.1), (0.29, 0.29), (0.31, 0.31)), 0.3)

        self.assertEqual(sorted(grid), [(-1, -1), (0, 0), (1, 1)])


class RegistrationDeltaTests(unittest.TestCase):
    def test_reanchor_matches_fresh_registration(self):
        from digital_twin.phone_mount import (
            host_from_mat4,
            registration_delta,
            rotation_z,
        )

        # A planar T_align (45 deg about twin Z + translation), as Phase 3 would
        # produce. Re-anchoring an identity-registered anchor by the delta must
        # match processing the same frame fresh under the new registration.
        t_align = rotation_z(math.radians(45.0), (0.3, -0.4, 0.0))
        new_registration = host_from_mat4(t_align)

        frame = LiveMeshAnchorFrame(
            anchor_id=str(uuid.uuid4()),
            timestamp_s=1.0,
            anchor_transform=identity_matrix4(),
            vertices=((0.2, 0.0, 0.1), (1.0, 0.5, 0.0), (-0.3, 0.7, 0.2)),
            faces=((0, 1, 2),),
            colors=((1, 2, 3), (4, 5, 6), (7, 8, 9)),
            normals=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        )

        old_state = LiveScanState(arkit_to_twin=identity_matrix4())
        old_state.apply_frame(frame, now_s=1.0)
        delta = registration_delta(identity_matrix4(), t_align)
        old_state.apply_registration_delta(delta, new_registration)

        fresh_state = LiveScanState(arkit_to_twin=new_registration)
        fresh_state.apply_frame(frame, now_s=1.0)

        anchor_id = frame.anchor_id
        reanchored = old_state.anchors[anchor_id].vertices_twin
        fresh = fresh_state.anchors[anchor_id].vertices_twin
        for a, b in zip(reanchored, fresh):
            for ca, cb in zip(a, b):
                self.assertAlmostEqual(ca, cb, places=5)
        self.assertEqual(old_state.arkit_to_twin, new_registration)


class PhonePoseProtocolTests(unittest.TestCase):
    def test_phone_pose_round_trip(self):
        # Exactly float32-representable values so the round-trip is bit-exact.
        transform = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.5, 0.0, 1.25, 1.0,
        )
        encoded = encode_phone_pose(
            transform,
            tracking_state=TRACKING_NORMAL,
            sequence=42,
            timestamp_s=98.25,
        )
        payload_length = struct.unpack_from("<I", encoded)[0]
        self.assertEqual(payload_length, len(encoded) - 4)

        frame = decode_payload(encoded[4:])
        self.assertTrue(frame.is_phone_pose)
        self.assertEqual(frame.message_type, MSG_PHONE_POSE)
        self.assertEqual(frame.anchor_transform, transform)
        self.assertEqual(frame.tracking_state, TRACKING_NORMAL)
        self.assertEqual(frame.sequence, 42)
        self.assertAlmostEqual(frame.timestamp_s, 98.25)

    def test_phone_pose_rejects_bad_length(self):
        encoded = encode_phone_pose(identity_matrix4())
        # Drop the trailing sequence uint32 from the body -> length mismatch.
        body = encoded[4:-4]
        truncated = struct.pack("<I", len(body)) + body
        with self.assertRaises(LiveScanProtocolError):
            decode_payload(truncated[4:])

    def test_phone_pose_requires_16_floats(self):
        with self.assertRaises(ValueError):
            encode_phone_pose((1.0, 0.0, 0.0))


class PhonePoseStoreTests(unittest.TestCase):
    def _frame(self, *, sequence=0, timestamp_s=0.0, tracking_state=TRACKING_NORMAL):
        return decode_payload(
            encode_phone_pose(
                identity_matrix4(),
                tracking_state=tracking_state,
                sequence=sequence,
                timestamp_s=timestamp_s,
            )[4:]
        )

    def test_stores_latest_and_counts(self):
        store = PhonePoseStore()
        self.assertIsNone(store.latest)
        store.update(self._frame(sequence=1, timestamp_s=10.0), received_monotonic_s=100.0)
        store.update(self._frame(sequence=2, timestamp_s=10.1), received_monotonic_s=100.1)
        self.assertIsNotNone(store.latest)
        self.assertEqual(store.latest.sequence, 2)
        self.assertEqual(store.frames_received, 2)
        self.assertEqual(store.sequence_gaps, 0)

    def test_clock_offset_set_once_against_host_monotonic(self):
        store = PhonePoseStore()
        store.update(self._frame(sequence=1, timestamp_s=10.0), received_monotonic_s=100.0)
        # offset = host_monotonic - phone_ts = 90.0, fixed on the first sample.
        self.assertAlmostEqual(store.clock_offset_s, 90.0)
        store.update(self._frame(sequence=2, timestamp_s=999.0), received_monotonic_s=200.0)
        self.assertAlmostEqual(store.clock_offset_s, 90.0)
        # A later phone timestamp maps back into the host monotonic clock.
        self.assertAlmostEqual(store.host_time_for(11.0), 101.0)

    def test_detects_sequence_gaps(self):
        store = PhonePoseStore()
        store.update(self._frame(sequence=1), received_monotonic_s=1.0)
        store.update(self._frame(sequence=2), received_monotonic_s=2.0)
        store.update(self._frame(sequence=7), received_monotonic_s=3.0)  # gap
        self.assertEqual(store.sequence_gaps, 1)

    def test_status_reports_tracking_and_age(self):
        store = PhonePoseStore()
        store.update(
            self._frame(sequence=1, timestamp_s=10.0, tracking_state=TRACKING_LIMITED),
            received_monotonic_s=100.0,
        )
        status = store.status(now_s=100.5)
        self.assertTrue(status["have_pose"])
        self.assertFalse(status["tracking_normal"])
        self.assertEqual(status["tracking_state"], TRACKING_LIMITED)
        self.assertAlmostEqual(status["age_s"], 0.5)

    def test_update_rejects_non_phone_frame(self):
        store = PhonePoseStore()
        with self.assertRaises(ValueError):
            store.update(decode_payload(encode_finalize()[4:]))


class PhonePoseReceiverIntegrationTests(unittest.TestCase):
    def _unused_port(self) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.close()
        return port

    def _connect_with_retry(self, port: int) -> socket.socket:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                return socket.create_connection(("127.0.0.1", port), timeout=0.5)
            except OSError:
                time.sleep(0.05)
        self.fail("could not connect to the receiver")

    def _read_payload(self, sock: socket.socket) -> bytes:
        length_data = sock.recv(4)
        self.assertEqual(len(length_data), 4)
        length = struct.unpack("<I", length_data)[0]
        payload = bytearray()
        while len(payload) < length:
            chunk = sock.recv(length - len(payload))
            self.assertTrue(chunk)
            payload.extend(chunk)
        return bytes(payload)

    def test_receiver_forwards_phone_pose_into_store(self):
        out_queue: queue.Queue = queue.Queue()
        receiver = LiveScanReceiver("127.0.0.1", 0, out_queue)
        # Bind to an ephemeral port ourselves so the test is deterministic.
        port = self._unused_port()
        receiver.port = port
        receiver.start()
        try:
            with self._connect_with_retry(port) as client:
                client.sendall(
                    encode_phone_pose(
                        identity_matrix4(),
                        tracking_state=TRACKING_NORMAL,
                        sequence=3,
                        timestamp_s=5.0,
                    )
                )
                frame = out_queue.get(timeout=2.0)
            self.assertTrue(frame.is_phone_pose)
            store = PhonePoseStore()
            store.update(frame, received_monotonic_s=10.0)
            self.assertEqual(store.latest.sequence, 3)
            self.assertAlmostEqual(store.clock_offset_s, 5.0)
        finally:
            receiver.stop()

    def test_idle_connection_survives_recv_timeouts(self):
        # A quiet gap between messages (e.g. the scan paused, only the slower pose
        # cadence in flight) must not drop the phone: the receiver keeps the
        # connection and still delivers a frame sent after several timeout windows.
        out_queue: queue.Queue = queue.Queue()
        receiver = LiveScanReceiver("127.0.0.1", 0, out_queue, read_timeout_s=0.05)
        port = self._unused_port()
        receiver.port = port
        receiver.start()
        try:
            with self._connect_with_retry(port) as client:
                time.sleep(0.3)  # span several read_timeout_s windows while idle
                client.sendall(
                    encode_phone_pose(
                        identity_matrix4(),
                        tracking_state=TRACKING_NORMAL,
                        sequence=7,
                        timestamp_s=1.0,
                    )
                )
                frame = out_queue.get(timeout=2.0)
            self.assertTrue(frame.is_phone_pose)
            self.assertEqual(frame.sequence, 7)
        finally:
            receiver.stop()

    def test_receiver_sends_initial_control_messages_on_connect(self):
        out_queue: queue.Queue = queue.Queue()
        receiver = LiveScanReceiver(
            "127.0.0.1",
            0,
            out_queue,
            initial_control_messages=[MSG_CONTROL_ENABLE_IMU],
        )
        port = self._unused_port()
        receiver.port = port
        receiver.start()
        try:
            with self._connect_with_retry(port) as client:
                client.settimeout(2.0)
                frame = decode_payload(self._read_payload(client))
            self.assertEqual(frame.message_type, MSG_CONTROL_ENABLE_IMU)
        finally:
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
