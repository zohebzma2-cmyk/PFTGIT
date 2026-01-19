import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  FlatList,
  Alert,
  Dimensions,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import * as Sharing from 'expo-sharing';
import Svg, { Path, Circle, Rect } from 'react-native-svg';
import { useProjectStore, Project, MediaItem, exportFunscript } from '../../store/projectStore';
import { useEditorStore } from '../../store/editorStore';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import colors from '../../constants/colors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = (SCREEN_WIDTH - 48) / 2;

// Icons
function FolderIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </Svg>
  );
}

function FileIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <Path d="M14 2v6h6" />
      <Path d="M16 13H8" />
      <Path d="M16 17H8" />
      <Path d="M10 9H8" />
    </Svg>
  );
}

function VideoIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Rect x={2} y={2} width={20} height={20} rx={2.18} ry={2.18} />
      <Path d="M7 2v20" />
      <Path d="M17 2v20" />
      <Path d="M2 12h20" />
      <Path d="M2 7h5" />
      <Path d="M2 17h5" />
      <Path d="M17 17h5" />
      <Path d="M17 7h5" />
    </Svg>
  );
}

function CheckCircleIcon({ color = colors.status.success, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <Path d="M22 4L12 14.01l-3-3" />
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

function TrashIcon({ color = colors.status.error, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M3 6h18" />
      <Path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </Svg>
  );
}

function ShareIcon({ color = colors.primary.DEFAULT, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
      <Path d="M16 6l-4-4-4 4" />
      <Path d="M12 2v13" />
    </Svg>
  );
}

function MoreIcon({ color = colors.text.muted, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Circle cx={12} cy={12} r={1} />
      <Circle cx={19} cy={12} r={1} />
      <Circle cx={5} cy={12} r={1} />
    </Svg>
  );
}

type TabType = 'drafts' | 'finished' | 'media';

function TabButton({
  title,
  active,
  onPress,
  count
}: {
  title: string;
  active: boolean;
  onPress: () => void;
  count?: number;
}) {
  return (
    <TouchableOpacity
      style={[styles.tabButton, active && styles.tabButtonActive]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <Text style={[styles.tabButtonText, active && styles.tabButtonTextActive]}>
        {title}
      </Text>
      {count !== undefined && count > 0 && (
        <View style={[styles.tabBadge, active && styles.tabBadgeActive]}>
          <Text style={[styles.tabBadgeText, active && styles.tabBadgeTextActive]}>
            {count}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

function ProjectCard({
  project,
  onPress,
  onDelete,
  onExport,
}: {
  project: Project;
  onPress: () => void;
  onDelete: () => void;
  onExport: () => void;
}) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <TouchableOpacity style={styles.projectCard} onPress={onPress} activeOpacity={0.8}>
      <View style={styles.projectThumbnail}>
        {project.status === 'finished' ? (
          <CheckCircleIcon color={colors.status.success} size={32} />
        ) : (
          <FileIcon color={colors.text.muted} size={32} />
        )}
      </View>
      <View style={styles.projectInfo}>
        <Text style={styles.projectName} numberOfLines={1}>
          {project.name}
        </Text>
        <Text style={styles.projectMeta}>
          {project.points.length} points • {formatDuration(project.duration)}
        </Text>
        <Text style={styles.projectDate}>{formatDate(project.updatedAt)}</Text>
      </View>
      <View style={styles.projectActions}>
        <TouchableOpacity onPress={onExport} style={styles.actionButton}>
          <ShareIcon />
        </TouchableOpacity>
        <TouchableOpacity onPress={onDelete} style={styles.actionButton}>
          <TrashIcon />
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
}

function MediaCard({
  media,
  onPress,
  onDelete
}: {
  media: MediaItem;
  onPress: () => void;
  onDelete: () => void;
}) {
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <TouchableOpacity style={styles.mediaCard} onPress={onPress} activeOpacity={0.8}>
      <View style={styles.mediaThumbnail}>
        <VideoIcon color={colors.text.muted} size={32} />
      </View>
      <Text style={styles.mediaName} numberOfLines={1}>
        {media.filename}
      </Text>
      <Text style={styles.mediaDuration}>{formatDuration(media.duration)}</Text>
      <TouchableOpacity
        onPress={onDelete}
        style={styles.mediaDeleteButton}
        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
      >
        <TrashIcon size={16} />
      </TouchableOpacity>
    </TouchableOpacity>
  );
}

function EmptyState({
  icon: Icon,
  title,
  subtitle,
  actionTitle,
  onAction
}: {
  icon: React.FC<{ color: string; size: number }>;
  title: string;
  subtitle: string;
  actionTitle?: string;
  onAction?: () => void;
}) {
  return (
    <View style={styles.emptyState}>
      <Icon color={colors.text.muted} size={64} />
      <Text style={styles.emptyTitle}>{title}</Text>
      <Text style={styles.emptySubtitle}>{subtitle}</Text>
      {actionTitle && onAction && (
        <Button
          title={actionTitle}
          onPress={onAction}
          icon={<PlusIcon color={colors.bg.base} size={20} />}
          style={{ marginTop: 16 }}
        />
      )}
    </View>
  );
}

export default function LibraryScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabType>('drafts');
  const [refreshing, setRefreshing] = useState(false);

  const {
    projects,
    mediaLibrary,
    getDrafts,
    getFinishedProjects,
    deleteProject,
    addMedia,
    removeMedia,
    createProject,
    setCurrentProject,
  } = useProjectStore();

  const { setVideo, setPoints } = useEditorStore();

  const drafts = getDrafts();
  const finished = getFinishedProjects();

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    // Simulate refresh
    setTimeout(() => setRefreshing(false), 500);
  }, []);

  const handleAddMedia = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please allow access to your media library.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      const media: MediaItem = {
        id: `media-${Date.now()}`,
        uri: asset.uri,
        filename: asset.fileName || 'video.mp4',
        mimeType: asset.mimeType || 'video/mp4',
        duration: (asset.duration || 0) * 1000,
        width: asset.width || 0,
        height: asset.height || 0,
        createdAt: new Date().toISOString(),
      };

      addMedia(media);
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

      Alert.alert(
        'Video Added',
        'Would you like to create a new project with this video?',
        [
          { text: 'Later', style: 'cancel' },
          {
            text: 'Create Project',
            onPress: () => handleCreateProjectFromMedia(media),
          },
        ]
      );
    }
  };

  const handleCreateProjectFromMedia = (media: MediaItem) => {
    const name = media.filename.replace(/\.[^/.]+$/, '');
    const project = createProject(name, media);

    // Load into editor
    setVideo(
      {
        id: media.id,
        filename: media.filename,
        originalName: media.filename,
        mimeType: media.mimeType,
        size: 0,
        duration: media.duration,
        width: media.width,
        height: media.height,
        fps: 30,
        uploadedAt: media.createdAt,
      },
      media.uri
    );

    router.push('/(tabs)/editor');
  };

  const handleOpenProject = (project: Project) => {
    setCurrentProject(project.id);

    if (project.media) {
      setVideo(
        {
          id: project.media.id,
          filename: project.media.filename,
          originalName: project.media.filename,
          mimeType: project.media.mimeType,
          size: 0,
          duration: project.media.duration,
          width: project.media.width,
          height: project.media.height,
          fps: 30,
          uploadedAt: project.media.createdAt,
        },
        project.media.uri
      );
    }

    setPoints(project.points);
    router.push('/(tabs)/editor');
  };

  const handleDeleteProject = (project: Project) => {
    Alert.alert(
      'Delete Project',
      `Are you sure you want to delete "${project.name}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            deleteProject(project.id);
            await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          },
        },
      ]
    );
  };

  const handleExportProject = async (project: Project) => {
    try {
      const funscriptJson = exportFunscript(project);

      // Show export options
      Alert.alert(
        'Export Funscript',
        `Export "${project.name}" with ${project.points.length} points?`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Copy to Clipboard',
            onPress: async () => {
              // Use Clipboard API if available
              if (typeof navigator !== 'undefined' && navigator.clipboard) {
                await navigator.clipboard.writeText(funscriptJson);
                Alert.alert('Copied!', 'Funscript copied to clipboard');
              } else {
                Alert.alert('Export Data', funscriptJson.substring(0, 500) + '...');
              }
              await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            },
          },
        ]
      );
    } catch (error) {
      Alert.alert('Export Failed', 'Could not export the funscript.');
    }
  };

  const handleDeleteMedia = (media: MediaItem) => {
    Alert.alert(
      'Delete Video',
      `Are you sure you want to remove "${media.filename}" from your library?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            removeMedia(media.id);
            await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          },
        },
      ]
    );
  };

  const handleNewProject = () => {
    Alert.prompt(
      'New Project',
      'Enter a name for your project:',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Create',
          onPress: (name: string | undefined) => {
            if (name?.trim()) {
              const project = createProject(name.trim());
              setCurrentProject(project.id);
              router.push('/(tabs)/editor');
            }
          },
        },
      ],
      'plain-text',
      'Untitled Project'
    );
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'drafts':
        if (drafts.length === 0) {
          return (
            <EmptyState
              icon={FolderIcon}
              title="No Drafts"
              subtitle="Your work-in-progress projects will appear here"
              actionTitle="New Project"
              onAction={handleNewProject}
            />
          );
        }
        return (
          <FlatList
            data={drafts}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <ProjectCard
                project={item}
                onPress={() => handleOpenProject(item)}
                onDelete={() => handleDeleteProject(item)}
                onExport={() => handleExportProject(item)}
              />
            )}
            contentContainerStyle={styles.listContent}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
            }
          />
        );

      case 'finished':
        if (finished.length === 0) {
          return (
            <EmptyState
              icon={CheckCircleIcon}
              title="No Finished Scripts"
              subtitle="Completed funscripts will appear here"
            />
          );
        }
        return (
          <FlatList
            data={finished}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <ProjectCard
                project={item}
                onPress={() => handleOpenProject(item)}
                onDelete={() => handleDeleteProject(item)}
                onExport={() => handleExportProject(item)}
              />
            )}
            contentContainerStyle={styles.listContent}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
            }
          />
        );

      case 'media':
        if (mediaLibrary.length === 0) {
          return (
            <EmptyState
              icon={VideoIcon}
              title="No Videos"
              subtitle="Add videos to your library to create funscripts"
              actionTitle="Add Video"
              onAction={handleAddMedia}
            />
          );
        }
        return (
          <FlatList
            data={mediaLibrary}
            keyExtractor={(item) => item.id}
            numColumns={2}
            columnWrapperStyle={styles.mediaGrid}
            renderItem={({ item }) => (
              <MediaCard
                media={item}
                onPress={() => handleCreateProjectFromMedia(item)}
                onDelete={() => handleDeleteMedia(item)}
              />
            )}
            contentContainerStyle={styles.listContent}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
            }
          />
        );
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Library</Text>
        <TouchableOpacity
          style={styles.addButton}
          onPress={activeTab === 'media' ? handleAddMedia : handleNewProject}
        >
          <PlusIcon color={colors.primary.DEFAULT} size={24} />
        </TouchableOpacity>
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        <TabButton
          title="Drafts"
          active={activeTab === 'drafts'}
          onPress={() => setActiveTab('drafts')}
          count={drafts.length}
        />
        <TabButton
          title="Finished"
          active={activeTab === 'finished'}
          onPress={() => setActiveTab('finished')}
          count={finished.length}
        />
        <TabButton
          title="Media"
          active={activeTab === 'media'}
          onPress={() => setActiveTab('media')}
          count={mediaLibrary.length}
        />
      </View>

      {/* Content */}
      <View style={styles.content}>{renderContent()}</View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.base,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: colors.text.primary,
  },
  addButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primary.dim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabBar: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    gap: 8,
    marginBottom: 16,
  },
  tabButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: colors.bg.elevated,
    gap: 6,
  },
  tabButtonActive: {
    backgroundColor: colors.primary.DEFAULT,
  },
  tabButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.secondary,
  },
  tabButtonTextActive: {
    color: colors.bg.base,
  },
  tabBadge: {
    backgroundColor: colors.bg.surface,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
    minWidth: 20,
    alignItems: 'center',
  },
  tabBadgeActive: {
    backgroundColor: 'rgba(0,0,0,0.2)',
  },
  tabBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.muted,
  },
  tabBadgeTextActive: {
    color: colors.bg.base,
  },
  content: {
    flex: 1,
  },
  listContent: {
    padding: 16,
    gap: 12,
  },
  projectCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    padding: 12,
    gap: 12,
  },
  projectThumbnail: {
    width: 56,
    height: 56,
    borderRadius: 8,
    backgroundColor: colors.bg.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  projectInfo: {
    flex: 1,
  },
  projectName: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 2,
  },
  projectMeta: {
    fontSize: 13,
    color: colors.text.secondary,
    marginBottom: 2,
  },
  projectDate: {
    fontSize: 12,
    color: colors.text.muted,
  },
  projectActions: {
    flexDirection: 'row',
    gap: 4,
  },
  actionButton: {
    padding: 8,
  },
  mediaGrid: {
    gap: 12,
  },
  mediaCard: {
    width: CARD_WIDTH,
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    overflow: 'hidden',
  },
  mediaThumbnail: {
    height: 100,
    backgroundColor: colors.bg.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  mediaName: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.primary,
    paddingHorizontal: 12,
    paddingTop: 10,
  },
  mediaDuration: {
    fontSize: 12,
    color: colors.text.muted,
    paddingHorizontal: 12,
    paddingBottom: 10,
    marginTop: 2,
  },
  mediaDeleteButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    padding: 4,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
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
    textAlign: 'center',
  },
});
