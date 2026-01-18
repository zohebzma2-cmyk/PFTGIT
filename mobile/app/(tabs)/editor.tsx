import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Alert,
} from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import Svg, { Path, Line, Circle, Polyline } from 'react-native-svg';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useEditorStore } from '../../store/editorStore';
import { useDeviceStore, syncFunscriptToDevice } from '../../store/deviceStore';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { DeviceSelector } from '../../components/DeviceSelector';
import { getDeviceFeatureDescription } from '../../data/deviceDatabase';
import colors from '../../constants/colors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const TIMELINE_HEIGHT = 150;

// Icons
function PlayIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <Path d="M8 5v14l11-7z" />
    </Svg>
  );
}

function PauseIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <Path d="M6 4h4v16H6zM14 4h4v16h-4z" />
    </Svg>
  );
}

function UploadIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <Path d="M17 8l-5-5-5 5" />
      <Path d="M12 3v12" />
    </Svg>
  );
}

function BluetoothIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M6.5 6.5l11 11L12 23V1l5.5 5.5-11 11" />
    </Svg>
  );
}

function PlusIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M12 5v14M5 12h14" />
    </Svg>
  );
}

function ZapIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
    </Svg>
  );
}

function Timeline({
  points,
  currentTime,
  duration,
  width,
}: {
  points: { at: number; pos: number }[];
  currentTime: number;
  duration: number;
  width: number;
}) {
  if (duration === 0) return null;

  const timeToX = (time: number) => (time / duration) * width;
  const posToY = (pos: number) => TIMELINE_HEIGHT - (pos / 100) * TIMELINE_HEIGHT;

  // Create path string for the funscript curve
  const pathData = points.length > 0
    ? points.map((p, i) => {
        const x = timeToX(p.at);
        const y = posToY(p.pos);
        return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
      }).join(' ')
    : '';

  return (
    <View style={styles.timelineContainer}>
      <Svg width={width} height={TIMELINE_HEIGHT}>
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((pos) => (
          <Line
            key={pos}
            x1={0}
            y1={posToY(pos)}
            x2={width}
            y2={posToY(pos)}
            stroke={colors.border.DEFAULT}
            strokeWidth={1}
            strokeDasharray={pos === 50 ? '' : '4,4'}
          />
        ))}

        {/* Funscript curve */}
        {pathData && (
          <Path
            d={pathData}
            stroke={colors.primary.DEFAULT}
            strokeWidth={2}
            fill="none"
          />
        )}

        {/* Points */}
        {points.map((point, index) => (
          <Circle
            key={index}
            cx={timeToX(point.at)}
            cy={posToY(point.pos)}
            r={6}
            fill={colors.primary.DEFAULT}
          />
        ))}

        {/* Playhead */}
        <Line
          x1={timeToX(currentTime)}
          y1={0}
          x2={timeToX(currentTime)}
          y2={TIMELINE_HEIGHT}
          stroke={colors.text.primary}
          strokeWidth={2}
        />
      </Svg>
    </View>
  );
}

export default function EditorScreen() {
  const videoRef = useRef<Video>(null);
  const [deviceSelectorVisible, setDeviceSelectorVisible] = useState(false);
  const [syncEnabled, setSyncEnabled] = useState(false);

  const {
    videoUri,
    points,
    currentTime,
    duration,
    isPlaying,
    isProcessing,
    processingProgress,
    setVideo,
    setCurrentTime,
    setDuration,
    setIsPlaying,
    addPoint,
    setProcessing,
  } = useEditorStore();

  const { connectedDevice, setConnectedDevice, disconnect } = useDeviceStore();

  // Sync funscript to device during playback
  useEffect(() => {
    if (syncEnabled && isPlaying && connectedDevice && points.length > 0) {
      const timestamps = points.map(p => p.at);
      const positions = points.map(p => p.pos);
      syncFunscriptToDevice(timestamps, positions, currentTime);
    }
  }, [syncEnabled, isPlaying, connectedDevice, points, currentTime]);

  const handlePickVideo = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please allow access to your media library to select videos.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      setVideo(
        {
          id: Date.now().toString(),
          filename: asset.fileName || 'video.mp4',
          originalName: asset.fileName || 'video.mp4',
          mimeType: asset.mimeType || 'video/mp4',
          size: asset.fileSize || 0,
          duration: (asset.duration || 0) * 1000,
          width: asset.width || 0,
          height: asset.height || 0,
          fps: 30,
          uploadedAt: new Date().toISOString(),
        },
        asset.uri
      );
    }
  };

  const handleRecordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please allow access to your camera to record videos.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 600, // 10 minutes max
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      setVideo(
        {
          id: Date.now().toString(),
          filename: asset.fileName || 'recorded.mp4',
          originalName: asset.fileName || 'recorded.mp4',
          mimeType: asset.mimeType || 'video/mp4',
          size: asset.fileSize || 0,
          duration: (asset.duration || 0) * 1000,
          width: asset.width || 0,
          height: asset.height || 0,
          fps: 30,
          uploadedAt: new Date().toISOString(),
        },
        asset.uri
      );
    }
  };

  const handlePlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    if (status.isLoaded) {
      setCurrentTime(status.positionMillis);
      setDuration(status.durationMillis || 0);
      setIsPlaying(status.isPlaying);
    }
  }, [setCurrentTime, setDuration, setIsPlaying]);

  const togglePlayPause = async () => {
    if (videoRef.current) {
      if (isPlaying) {
        await videoRef.current.pauseAsync();
      } else {
        await videoRef.current.playAsync();
      }
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };

  const handleAddPoint = async () => {
    addPoint({ at: currentTime, pos: 50 });
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  };

  const handleGenerate = async () => {
    if (!videoUri) {
      Alert.alert('No Video', 'Please select a video first');
      return;
    }

    setProcessing(true, 0);

    // Simulate AI generation (in production, this would call the API)
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 500));
      setProcessing(true, i);
    }

    // Generate sample points
    const samplePoints = [];
    for (let t = 0; t < duration; t += 2000) {
      samplePoints.push({
        at: t,
        pos: Math.floor(Math.random() * 100),
      });
    }

    useEditorStore.getState().setPoints(samplePoints);
    setProcessing(false);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (!videoUri) {
    return (
      <SafeAreaView style={styles.container} edges={['bottom']}>
        <View style={styles.emptyState}>
          <UploadIcon size={64} color={colors.text.muted} />
          <Text style={styles.emptyTitle}>No video loaded</Text>
          <Text style={styles.emptySubtitle}>
            Select a video from your library or record a new one
          </Text>
          <View style={styles.uploadButtons}>
            <Button
              title="Choose Video"
              onPress={handlePickVideo}
              icon={<UploadIcon color={colors.bg.base} size={20} />}
            />
            <Button
              title="Record Video"
              variant="outline"
              onPress={handleRecordVideo}
            />
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        {/* Video Player */}
        <View style={styles.videoContainer}>
          <Video
            ref={videoRef}
            source={{ uri: videoUri }}
            style={styles.video}
            resizeMode={ResizeMode.CONTAIN}
            onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
            shouldPlay={false}
          />
        </View>

        {/* Playback Controls */}
        <View style={styles.controls}>
          <TouchableOpacity onPress={togglePlayPause} style={styles.playButton}>
            {isPlaying ? (
              <PauseIcon color={colors.text.primary} size={32} />
            ) : (
              <PlayIcon color={colors.text.primary} size={32} />
            )}
          </TouchableOpacity>
          <Text style={styles.timeText}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </Text>
        </View>

        {/* Timeline */}
        <Card style={styles.timelineCard}>
          <Text style={styles.sectionTitle}>Timeline</Text>
          <Timeline
            points={points}
            currentTime={currentTime}
            duration={duration}
            width={SCREEN_WIDTH - 64}
          />
          <View style={styles.timelineActions}>
            <Button
              title="Add Point"
              size="sm"
              variant="outline"
              onPress={handleAddPoint}
            />
            <Text style={styles.pointCount}>{points.length} points</Text>
          </View>
        </Card>

        {/* AI Generation */}
        <Card style={styles.generateCard}>
          <Text style={styles.sectionTitle}>AI Generation</Text>
          {isProcessing ? (
            <View style={styles.processingContainer}>
              <View style={styles.progressBar}>
                <View
                  style={[styles.progressFill, { width: `${processingProgress}%` }]}
                />
              </View>
              <Text style={styles.progressText}>
                Processing... {processingProgress}%
              </Text>
            </View>
          ) : (
            <Button
              title="Generate Funscript"
              onPress={handleGenerate}
              fullWidth
            />
          )}
        </Card>

        {/* Statistics */}
        <Card style={styles.statsCard}>
          <Text style={styles.sectionTitle}>Statistics</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{points.length}</Text>
              <Text style={styles.statLabel}>Points</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{formatTime(duration)}</Text>
              <Text style={styles.statLabel}>Duration</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>
                {points.length > 1
                  ? Math.round(
                      points.reduce((acc, p, i, arr) => {
                        if (i === 0) return 0;
                        return acc + Math.abs(p.pos - arr[i - 1].pos);
                      }, 0) / points.length
                    )
                  : 0}
              </Text>
              <Text style={styles.statLabel}>Avg Speed</Text>
            </View>
          </View>
        </Card>

        {/* Device Status */}
        <Card style={styles.deviceCard}>
          <Text style={styles.sectionTitle}>Device</Text>
          {connectedDevice ? (
            <View style={styles.deviceContent}>
              <View style={styles.deviceInfo}>
                <View style={styles.deviceHeader}>
                  <BluetoothIcon color={colors.status.success} size={20} />
                  <Text style={styles.deviceName}>{connectedDevice.name}</Text>
                  <View style={styles.connectedBadge}>
                    <Text style={styles.connectedBadgeText}>Connected</Text>
                  </View>
                </View>
                <Text style={styles.deviceDetails}>
                  {connectedDevice.manufacturer} • {getDeviceFeatureDescription(connectedDevice.features)}
                </Text>
                {connectedDevice.batteryLevel !== undefined && (
                  <Text style={styles.deviceBattery}>
                    Battery: {connectedDevice.batteryLevel}%
                  </Text>
                )}
              </View>
              <View style={styles.deviceActions}>
                <TouchableOpacity
                  style={[styles.syncButton, syncEnabled && styles.syncButtonActive]}
                  onPress={() => setSyncEnabled(!syncEnabled)}
                >
                  <ZapIcon color={syncEnabled ? colors.bg.base : colors.primary.DEFAULT} size={18} />
                  <Text style={[styles.syncButtonText, syncEnabled && styles.syncButtonTextActive]}>
                    {syncEnabled ? 'Syncing' : 'Sync'}
                  </Text>
                </TouchableOpacity>
                <Button
                  title="Change"
                  size="sm"
                  variant="outline"
                  onPress={() => setDeviceSelectorVisible(true)}
                />
              </View>
            </View>
          ) : (
            <TouchableOpacity
              style={styles.addDeviceButton}
              onPress={() => setDeviceSelectorVisible(true)}
            >
              <View style={styles.addDeviceIcon}>
                <PlusIcon color={colors.primary.DEFAULT} size={24} />
              </View>
              <View style={styles.addDeviceText}>
                <Text style={styles.addDeviceTitle}>Add Device</Text>
                <Text style={styles.addDeviceSubtitle}>
                  Connect Lovense, Kiiroo, We-Vibe & more
                </Text>
              </View>
            </TouchableOpacity>
          )}
        </Card>
      </ScrollView>

      <DeviceSelector
        visible={deviceSelectorVisible}
        onClose={() => setDeviceSelectorVisible(false)}
        onDeviceConnected={setConnectedDevice}
        connectedDevice={connectedDevice}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.base,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
    gap: 16,
  },
  videoContainer: {
    aspectRatio: 16 / 9,
    backgroundColor: colors.bg.surface,
    borderRadius: 12,
    overflow: 'hidden',
  },
  video: {
    flex: 1,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    paddingHorizontal: 8,
  },
  playButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.bg.elevated,
    alignItems: 'center',
    justifyContent: 'center',
  },
  timeText: {
    fontSize: 14,
    color: colors.text.secondary,
    fontVariant: ['tabular-nums'],
  },
  timelineCard: {
    padding: 16,
  },
  timelineContainer: {
    marginTop: 8,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: colors.bg.elevated,
  },
  timelineActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  pointCount: {
    fontSize: 14,
    color: colors.text.muted,
  },
  generateCard: {
    padding: 16,
  },
  processingContainer: {
    gap: 8,
  },
  progressBar: {
    height: 8,
    backgroundColor: colors.bg.elevated,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary.DEFAULT,
  },
  progressText: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  statsCard: {
    padding: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 8,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.primary.DEFAULT,
  },
  statLabel: {
    fontSize: 12,
    color: colors.text.muted,
    marginTop: 4,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 8,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.primary,
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 14,
    color: colors.text.secondary,
    marginBottom: 24,
    textAlign: 'center',
  },
  uploadButtons: {
    gap: 12,
  },
  deviceCard: {
    padding: 16,
  },
  deviceContent: {
    gap: 12,
  },
  deviceInfo: {
    gap: 4,
  },
  deviceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  deviceName: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    flex: 1,
  },
  connectedBadge: {
    backgroundColor: colors.status.success + '20',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  connectedBadgeText: {
    color: colors.status.success,
    fontSize: 11,
    fontWeight: '600',
  },
  deviceDetails: {
    fontSize: 13,
    color: colors.text.secondary,
    marginLeft: 28,
  },
  deviceBattery: {
    fontSize: 12,
    color: colors.text.muted,
    marginLeft: 28,
  },
  deviceActions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 4,
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.primary.DEFAULT,
  },
  syncButtonActive: {
    backgroundColor: colors.primary.DEFAULT,
  },
  syncButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary.DEFAULT,
  },
  syncButtonTextActive: {
    color: colors.bg.base,
  },
  addDeviceButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 12,
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border.DEFAULT,
    borderStyle: 'dashed',
  },
  addDeviceIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.primary.dim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  addDeviceText: {
    flex: 1,
  },
  addDeviceTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  addDeviceSubtitle: {
    fontSize: 13,
    color: colors.text.secondary,
    marginTop: 2,
  },
});
