import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import { useLocalSearchParams, useRouter, Stack } from 'expo-router';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { SafeAreaView } from 'react-native-safe-area-context';
import { projectsApi } from '../../api/client';
import { useEditorStore } from '../../store/editorStore';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import colors from '../../constants/colors';
import type { Project } from '../../api/types';

export default function ProjectDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const resetEditor = useEditorStore((state) => state.reset);

  const { data: project, isLoading, error } = useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.get(id!),
    enabled: !!id,
  });

  const deleteMutation = useMutation({
    mutationFn: () => projectsApi.delete(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      router.back();
    },
  });

  const handleOpenEditor = () => {
    resetEditor();
    router.push('/(tabs)/editor');
  };

  const handleDelete = () => {
    Alert.alert(
      'Delete Project',
      'Are you sure you want to delete this project? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteMutation.mutate(),
        },
      ]
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centered}>
          <Text style={styles.loadingText}>Loading project...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || !project) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centered}>
          <Text style={styles.errorText}>Project not found</Text>
          <Button title="Go Back" onPress={() => router.back()} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <>
      <Stack.Screen
        options={{
          title: project.name,
          headerRight: () => (
            <Button
              title="Delete"
              variant="ghost"
              size="sm"
              onPress={handleDelete}
              textStyle={{ color: colors.status.error }}
            />
          ),
        }}
      />
      <SafeAreaView style={styles.container} edges={['bottom']}>
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
          <Card style={styles.infoCard}>
            <Text style={styles.projectName}>{project.name}</Text>
            {project.description && (
              <Text style={styles.projectDescription}>{project.description}</Text>
            )}
            <Text style={styles.projectDate}>
              Created {new Date(project.createdAt).toLocaleDateString()}
            </Text>
          </Card>

          <Card style={styles.statsCard}>
            <Text style={styles.sectionTitle}>Project Stats</Text>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {project.videoId ? '1' : '0'}
                </Text>
                <Text style={styles.statLabel}>Videos</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {project.funscriptId ? '1' : '0'}
                </Text>
                <Text style={styles.statLabel}>Funscripts</Text>
              </View>
            </View>
          </Card>

          <View style={styles.actions}>
            <Button
              title="Open in Editor"
              onPress={handleOpenEditor}
              fullWidth
            />
            <Button
              title="Export Funscript"
              variant="outline"
              onPress={() => Alert.alert('Export', 'Export functionality coming soon')}
              fullWidth
              disabled={!project.funscriptId}
            />
          </View>
        </ScrollView>
      </SafeAreaView>
    </>
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
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  loadingText: {
    color: colors.text.secondary,
    fontSize: 16,
  },
  errorText: {
    color: colors.status.error,
    fontSize: 16,
  },
  infoCard: {
    padding: 20,
  },
  projectName: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 8,
  },
  projectDescription: {
    fontSize: 16,
    color: colors.text.secondary,
    marginBottom: 12,
  },
  projectDate: {
    fontSize: 14,
    color: colors.text.muted,
  },
  statsCard: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    marginBottom: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 32,
    fontWeight: '700',
    color: colors.primary.DEFAULT,
  },
  statLabel: {
    fontSize: 14,
    color: colors.text.muted,
    marginTop: 4,
  },
  actions: {
    gap: 12,
  },
});
