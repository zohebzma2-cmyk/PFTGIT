import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Path } from 'react-native-svg';
import { projectsApi } from '../../api/client';
import { useAuthStore } from '../../store/authStore';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import colors from '../../constants/colors';
import type { Project } from '../../api/types';

function PlusIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M12 5v14M5 12h14" />
    </Svg>
  );
}

function FolderIcon({ color = colors.primary.DEFAULT, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </Svg>
  );
}

function ProjectCard({ project, onPress }: { project: Project; onPress: () => void }) {
  const formattedDate = new Date(project.createdAt).toLocaleDateString();

  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
      <Card style={styles.projectCard}>
        <View style={styles.projectIcon}>
          <FolderIcon size={32} />
        </View>
        <View style={styles.projectInfo}>
          <Text style={styles.projectName}>{project.name}</Text>
          {project.description && (
            <Text style={styles.projectDescription} numberOfLines={2}>
              {project.description}
            </Text>
          )}
          <Text style={styles.projectDate}>Created {formattedDate}</Text>
        </View>
      </Card>
    </TouchableOpacity>
  );
}

export default function ProjectsScreen() {
  const router = useRouter();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  const {
    data: projects,
    isLoading,
    refetch,
    isRefetching,
  } = useQuery({
    queryKey: ['projects'],
    queryFn: projectsApi.list,
    enabled: isAuthenticated,
  });

  const handleCreateProject = async () => {
    try {
      const project = await projectsApi.create({
        name: `Project ${(projects?.length || 0) + 1}`,
      });
      router.push(`/project/${project.id}`);
    } catch (error) {
      console.error('Failed to create project:', error);
    }
  };

  const handleProjectPress = (project: Project) => {
    router.push(`/project/${project.id}`);
  };

  if (!isAuthenticated) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.authPrompt}>
          <Text style={styles.authTitle}>Welcome to FunGen</Text>
          <Text style={styles.authSubtitle}>
            Sign in to create and manage your projects
          </Text>
          <Button
            title="Sign In"
            onPress={() => router.push('/(auth)/login')}
            style={styles.authButton}
          />
          <Button
            title="Create Account"
            variant="outline"
            onPress={() => router.push('/(auth)/register')}
            style={styles.authButton}
          />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.header}>
        <Text style={styles.title}>Your Projects</Text>
        <TouchableOpacity onPress={handleCreateProject} style={styles.addButton}>
          <PlusIcon color={colors.primary.DEFAULT} />
        </TouchableOpacity>
      </View>

      <FlatList
        data={projects}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <ProjectCard project={item} onPress={() => handleProjectPress(item)} />
        )}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl
            refreshing={isRefetching}
            onRefresh={refetch}
            tintColor={colors.primary.DEFAULT}
          />
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <FolderIcon size={64} color={colors.text.muted} />
            <Text style={styles.emptyTitle}>No projects yet</Text>
            <Text style={styles.emptySubtitle}>
              Create your first project to get started
            </Text>
            <Button
              title="Create Project"
              onPress={handleCreateProject}
              icon={<PlusIcon color={colors.bg.base} size={20} />}
              style={styles.createButton}
            />
          </View>
        }
      />
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
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.text.primary,
  },
  addButton: {
    padding: 8,
    borderRadius: 8,
    backgroundColor: colors.bg.elevated,
  },
  list: {
    padding: 16,
    gap: 12,
  },
  projectCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 12,
  },
  projectIcon: {
    width: 56,
    height: 56,
    borderRadius: 12,
    backgroundColor: colors.primary.dim,
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
    marginBottom: 4,
  },
  projectDescription: {
    fontSize: 14,
    color: colors.text.secondary,
    marginBottom: 4,
  },
  projectDate: {
    fontSize: 12,
    color: colors.text.muted,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 80,
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
  createButton: {
    minWidth: 180,
  },
  authPrompt: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  authTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 8,
  },
  authSubtitle: {
    fontSize: 16,
    color: colors.text.secondary,
    marginBottom: 32,
    textAlign: 'center',
  },
  authButton: {
    minWidth: 200,
    marginBottom: 12,
  },
});
