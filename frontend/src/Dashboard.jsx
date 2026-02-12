import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import ForceGraph2D from 'react-force-graph-2d';
import {
  Folder, FileText, Activity, Share2, Terminal, Box, Cpu,
  ArrowRight, Database, Layers, Settings as SettingsIcon,
  ChevronRight, Search, Zap, Shield, Clock, Server, Key,
  HardDrive, FolderOpen, BarChart3, Upload, Trash2, Move,
  RefreshCw, Eye, Download, X, Plus, ChevronDown, AlertCircle,
  CheckCircle, FolderInput, Volume2, Loader, Square,
  Sun, Moon, Monitor, MessageSquare, GitBranch, Copy, FileDown,
  Thermometer, History, Sparkles, Send, VolumeX, Play
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { playClusterChime, playFileMoveSwish, playUploadPing, playCommandResponse, playDuplicateAlert } from './sounds';

const SEFS_API = "http://localhost:8000";

const getSystemTheme = () => window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
const getInitialTheme = () => {
  const saved = localStorage.getItem('sefs-theme');
  return saved || 'system';
};

const Dashboard = () => {
  const [theme, setTheme] = useState(getInitialTheme);
  const [data, setData] = useState({ nodes: [], links: [] });
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState({ folders: 0, files: 0, ops: 0 });
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState('browse');
  const [systemInfo, setSystemInfo] = useState(null);
  const [fileStructure, setFileStructure] = useState([]);
  const [backendStats, setBackendStats] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [loadingFile, setLoadingFile] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [showMoveModal, setShowMoveModal] = useState(null);
  const [newFolderName, setNewFolderName] = useState('');
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [expandedFolder, setExpandedFolder] = useState(null);
  const [actionFeedback, setActionFeedback] = useState(null);
  const [recategorizing, setRecategorizing] = useState(null);
  const [fileSummary, setFileSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [summaryRequested, setSummaryRequested] = useState(false);

  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);

  // Audio State
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const audioRef = useRef(null);

  // Graph sizing state
  const [graphDimensions, setGraphDimensions] = useState({ width: 0, height: 0 });

  const fgRef = useRef();
  const fileInputRef = useRef();
  const graphContainerRef = useRef(null);
  const mainRef = useRef(null);

  // Resizable sidebar state
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem('sefs-sidebar-width');
    return saved ? parseInt(saved, 10) : 384;
  });
  const isDragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(384);

  // Show temporary feedback
  const showFeedback = (message, type = 'success') => {
    setActionFeedback({ message, type });
    setTimeout(() => setActionFeedback(null), 3000);
  };

  // ─── Standout Feature State ─────────────────────────────────────
  const [nlCommand, setNlCommand] = useState('');
  const [nlResponse, setNlResponse] = useState(null);
  const [nlLoading, setNlLoading] = useState(false);
  const [showNlBar, setShowNlBar] = useState(false);

  // Local Scan State
  const [scanPath, setScanPath] = useState('');
  const [timelineSnapshots, setTimelineSnapshots] = useState([]);
  const [timelineIndex, setTimelineIndex] = useState(-1);
  const [showTimeline, setShowTimeline] = useState(false);
  const [duplicates, setDuplicates] = useState([]);
  const [crossEdges, setCrossEdges] = useState([]);
  const [entropyData, setEntropyData] = useState({});
  const [showEntropy, setShowEntropy] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(() => {
    const saved = localStorage.getItem('sefs-sound');
    return saved === null ? true : saved === 'true';
  });
  const [folderSummaries, setFolderSummaries] = useState({});
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const nlInputRef = useRef(null);

  // Debug: Log showUpload state changes
  useEffect(() => {
    console.log('showUpload state changed:', showUpload);
  }, [showUpload]);

  // Theme effect — apply data-theme attribute to <html>
  useEffect(() => {
    const applyTheme = (t) => {
      const resolved = t === 'system' ? getSystemTheme() : t;
      document.documentElement.setAttribute('data-theme', resolved);
    };
    applyTheme(theme);

    if (theme === 'system') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = () => applyTheme('system');
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }
  }, [theme]);

  // Resizable sidebar drag handlers
  const onResizeStart = useCallback((e) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartX.current = e.clientX;
    dragStartWidth.current = sidebarWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [sidebarWidth]);

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!isDragging.current) return;
      const delta = dragStartX.current - e.clientX;
      const newWidth = Math.min(600, Math.max(280, dragStartWidth.current + delta));
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        localStorage.setItem('sefs-sidebar-width', String(sidebarWidth));
      }
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [sidebarWidth]);

  // Data Fetching
  const fetchSystemInfo = useCallback(async () => {
    try {
      const res = await axios.get(SEFS_API + '/system-info');
      setSystemInfo(res.data);
      console.log('System info:', res.data);
    } catch (err) {
      console.error("Failed to fetch system info", err);
    }
  }, []);

  const fetchBackendStats = useCallback(async () => {
    try {
      const res = await axios.get(`${SEFS_API}/stats`);
      setBackendStats(res.data);
      console.log('Backend stats:', res.data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  }, []);

  useEffect(() => {
    fetchSystemInfo();
    fetchBackendStats();
    const interval = setInterval(() => {
      fetchSystemInfo();
      fetchBackendStats();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchSystemInfo, fetchBackendStats]);

  const fetchStructure = useCallback(async () => {
    try {
      const res = await axios.get(`${SEFS_API}/files`);
      setFileStructure(res.data);
      const nodes = [{ id: 'Root', name: 'SEFS Root', group: 'root', val: 12 }];
      const links = [];

      res.data.forEach((item) => {
        // Skip hidden folders/files
        if (item.name.startsWith('.')) return;

        if (item.type === 'folder') {
          nodes.push({ id: item.name, name: item.name, group: 'folder', val: 8 });
          links.push({ source: 'Root', target: item.name });
          (item.files || []).forEach((file) => {
            const fname = typeof file === 'string' ? file : file.name;
            if (fname.startsWith('.')) return; // Skip hidden files inside folders

            // Store folder name in node data for click handler
            nodes.push({ id: `${item.name}/${fname}`, name: fname, group: 'file', val: 4, folder: item.name });
            links.push({ source: item.name, target: `${item.name}/${fname}` });
          });
        } else {
          nodes.push({ id: item.name, name: item.name, group: 'file', val: 4, folder: null });
          links.push({ source: 'Root', target: item.name });
        }
      });

      setData({ nodes, links });
      setStats(prev => ({
        ...prev,
        folders: res.data.filter(i => i.type === 'folder').length,
        files: nodes.filter(n => n.group === 'file').length
      }));

      // Initial zoom fit
      if (!isLoaded && nodes.length > 0) {
        setIsLoaded(true);
        setTimeout(() => fgRef.current?.zoomToFit(400, 100), 500);
      }
    } catch (err) {
      console.error("Failed to fetch structure", err);
    }
  }, [isLoaded]);

  useEffect(() => {
    fetchStructure();
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      // Use backend timestamp if available, otherwise current seconds
      const timestamp = update.timestamp || (Date.now() / 1000);
      setLogs(prev => [{ ...update, timestamp }, ...prev].slice(0, 30));
      setStats(prev => ({ ...prev, ops: prev.ops + 1 }));
      fetchStructure();
      fetchBackendStats();
    };

    ws.onopen = async () => {
      try {
        const res = await axios.get(`${SEFS_API}/logs`);
        if (res.data) setLogs(res.data);
      } catch (err) {
        // /logs may not exist in upload-driven mode, silently ignore
      }

      setLogs(prev => [{
        type: 'system',
        message: 'Intelligence Bridge Active',
        timestamp: Date.now() / 1000,
        manual: true
      }, ...prev].slice(0, 30));
    };
    return () => ws.close();
  }, [fetchStructure, fetchBackendStats]);

  // Handle resizing for graph
  useEffect(() => {
    const updateDimensions = () => {
      if (graphContainerRef.current) {
        setGraphDimensions({
          width: graphContainerRef.current.clientWidth,
          height: graphContainerRef.current.clientHeight
        });
      }
    };

    updateDimensions(); // Initial measurement

    const observer = new ResizeObserver(() => {
      updateDimensions();
    });

    if (graphContainerRef.current) {
      observer.observe(graphContainerRef.current);
    }

    return () => observer.disconnect();
  }, [activeTab]); // Re-run when tab changes to ensure correct sizing

  // Configure d3 forces for tight clustering
  useEffect(() => {
    if (!fgRef.current) return;
    const fg = fgRef.current;

    // Shorter link distances — keeps files close to their folders
    fg.d3Force('link')?.distance(link => {
      const s = link.source;
      const t = link.target;
      const sGroup = typeof s === 'object' ? s.group : '';
      const tGroup = typeof t === 'object' ? t.group : '';
      if (sGroup === 'root' || tGroup === 'root') return 60;
      return 35;  // file↔folder tight
    });

    // Moderate repulsion to prevent overlap
    fg.d3Force('charge')?.strength(-100);

    // Mild center pull keeps everything from drifting
    fg.d3Force('center')?.strength(0.12);
  }, [data]);

  // ─── Real Actions ────────────────────────────────────────────────────────

  const viewFile = async (filename, folder) => {
    console.log('viewFile called for:', filename, folder);
    setLoadingFile(true);
    setSelectedFile({ name: filename, folder });
    setFileSummary(null);
    setShowSummary(false);
    setSummaryRequested(false);
    try {
      const encFilename = encodeURIComponent(filename);
      const url = folder
        ? `${SEFS_API}/file/${encodeURIComponent(folder)}/${encFilename}`
        : `${SEFS_API}/file-root/${encFilename}`;

      console.log('Fetching file from:', url);
      const res = await axios.get(url);
      setFileContent(res.data);
    } catch (err) {
      console.error('Failed to load file:', err);
      setFileContent({ error: err.response?.data?.detail || 'Failed to load file' });
    }
    setLoadingFile(false);
  };

  const handleRequestSummary = () => {
    if (selectedFile) {
      setSummaryRequested(true);
      setShowSummary(true);
      fetchFileSummary(selectedFile.name, selectedFile.folder);
    }
  };

  const fetchFileSummary = async (filename, folder) => {
    setLoadingSummary(true);
    try {
      const encFilename = encodeURIComponent(filename);
      const url = folder
        ? `${SEFS_API}/file-summary/${encFilename}?folder=${encodeURIComponent(folder)}`
        : `${SEFS_API}/file-summary/${encFilename}`;
      const res = await axios.get(url);
      setFileSummary(res.data.summary);
    } catch (err) {
      console.error('Failed to fetch summary:', err);
      setFileSummary(null);
    }
    setLoadingSummary(false);
  };

  const handleReadSummary = async () => {
    // Always stop existing audio first
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }

    // If we were just playing (toggle), then we are done
    if (audioPlaying) {
      setAudioPlaying(false);
      setAudioLoading(false);
      return;
    }

    setAudioLoading(true);
    setAudioPlaying(false);

    try {
      const token = Math.random().toString(36).substring(7);
      const url = `${SEFS_API}/read-summary/${encodeURIComponent(selectedFile.name)}?folder=${encodeURIComponent(selectedFile.folder || '')}&t=${token}`;

      const audio = new Audio(url);
      audioRef.current = audio;

      audio.oncanplaythrough = () => {
        // Only play if this audio is still the active one
        if (audioRef.current === audio) {
          setAudioLoading(false);
          setAudioPlaying(true);
          audio.play().catch(e => console.error("Playback failed", e));
        }
      };

      audio.onended = () => {
        if (audioRef.current === audio) {
          setAudioPlaying(false);
          setAudioLoading(false);
        }
      };

      audio.onerror = (e) => {
        if (audioRef.current === audio) {
          console.error("Audio error", e);
          setAudioLoading(false);
          setAudioPlaying(false);
          showFeedback('Failed to load audio', 'error');
        }
      };

    } catch (err) {
      console.error("Read summary error:", err);
      setAudioLoading(false);
      showFeedback('Failed to generate audio', 'error');
    }
  };

  const closeFileViewer = () => {
    setSelectedFile(null);
    setFileContent(null);
    setFileSummary(null);
    // Force stop audio and clear ref so callbacks don't fire
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    setAudioPlaying(false);
    setAudioLoading(false);
  };

  const uploadFile = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    console.log('Starting upload of', files.length, 'files');
    setUploading(true);
    setUploadResult(null);

    try {
      const uploadPromises = [];
      let totalFiles = 0;

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        console.log('Uploading file:', file.name);
        const formData = new FormData();
        formData.append('file', file);
        uploadPromises.push(
          axios.post(`${SEFS_API}/upload?auto_categorize=true`, formData)
        );
        totalFiles++;
      }

      const results = await Promise.all(uploadPromises);
      const successCount = results.filter(r => !r.data.error).length;

      console.log('Upload complete:', successCount, 'successful');

      setUploadResult({
        success: true,
        message: `${successCount}/${totalFiles} files uploaded successfully`
      });
      showFeedback(`Uploaded ${successCount} file(s)`);
      fetchStructure();

      // Auto-close modal after 2 seconds
      setTimeout(() => {
        setShowUpload(false);
        setUploadResult(null);
      }, 2000);

    } catch (err) {
      console.error('Upload error:', err);
      setUploadResult({ error: err.response?.data?.detail || err.message || 'Upload failed' });
      showFeedback('Upload failed', 'error');
    }

    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';

    // Reset folder input too
    const folderInput = document.getElementById('folder-upload');
    if (folderInput) folderInput.value = '';
  };

  const moveFile = async (filename, sourceFolder, targetFolder) => {
    try {
      await axios.post(`${SEFS_API}/move`, {
        filename,
        source_folder: sourceFolder,
        target_folder: targetFolder
      });
      showFeedback(`Moved "${filename}" → ${targetFolder}`, 'success');
      playAmbientSound('move');

      // Close both the move modal AND the file viewer
      setShowMoveModal(null);
      setSelectedFile(null);
      setFileContent(null);

      // Refresh graph after short delay for DB commit
      setTimeout(() => fetchStructure(), 300);
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Move failed', 'error');
    }
  };

  const deleteFile = async (filename, folder) => {
    if (!window.confirm(`Delete "${filename}" permanently?`)) return;
    try {
      await axios.post(`${SEFS_API}/delete-file`, { filename, folder });
      showFeedback(`Deleted "${filename}"`);
      closeFileViewer();
      fetchStructure();
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Delete failed', 'error');
    }
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return;
    setCreatingFolder(true);
    try {
      await axios.post(`${SEFS_API}/create-folder/${newFolderName.trim()}`);
      showFeedback(`Created folder "${newFolderName.trim()}"`);
      setNewFolderName('');
      fetchStructure();
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Failed to create folder', 'error');
    }
    setCreatingFolder(false);
  };

  const deleteFolder = async (folderName) => {
    if (!window.confirm(`Delete folder "${folderName}"? It must be empty.`)) return;
    try {
      await axios.delete(`${SEFS_API}/folder/${folderName}`);
      showFeedback(`Deleted folder "${folderName}"`);
      fetchStructure();
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Failed to delete folder', 'error');
    }
  };

  const clearAll = async () => {
    if (!window.confirm('⚠️ Clear ALL data? This will delete all files, embeddings, clusters, and FAISS index. This cannot be undone.')) return;
    try {
      await axios.post(`${SEFS_API}/clear`);
      showFeedback('All data cleared successfully');
      setFileStructure([]);
      setData({ nodes: [], links: [] });
      setStats({ folders: 0, files: 0, ops: 0 });
      setLogs([]);
      fetchStructure();
      fetchBackendStats();
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Clear failed', 'error');
    }
  };

  const reclusterFiles = async () => {
    try {
      showFeedback('Re-clustering files...');
      await axios.post(`${SEFS_API}/recluster`);
      showFeedback('Re-clustering complete');
      fetchStructure();
      fetchBackendStats();
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Re-cluster failed', 'error');
    }
  };

  const recategorizeFile = async (filename, folder) => {
    setRecategorizing(`${folder}/${filename}`);
    try {
      const res = await axios.post(`${SEFS_API}/recategorize`, { filename, folder });
      if (res.data.same) {
        showFeedback(`AI confirms "${filename}" belongs in ${folder}`);
      } else {
        if (window.confirm(`AI suggests moving "${filename}" to "${res.data.suggested_folder}". Move now?`)) {
          await moveFile(filename, folder, res.data.suggested_folder);
        } else {
          showFeedback(`Suggestion: ${res.data.suggested_folder} (not moved)`);
        }
      }
    } catch (err) {
      showFeedback(err.response?.data?.detail || 'Re-analysis failed', 'error');
    }
    setRecategorizing(null);
  };

  const downloadFile = (filename, folder) => {
    const url = folder
      ? `${SEFS_API}/download?filename=${encodeURIComponent(filename)}&folder=${encodeURIComponent(folder)}`
      : `${SEFS_API}/download?filename=${encodeURIComponent(filename)}`;
    window.open(url, '_blank');
  };

  const handleNodeClick = useCallback((node) => {
    if (node.group === 'file') {
      viewFile(node.name, node.folder);
    } else {
      // For folders, zoom in
      fgRef.current?.centerAt(node.x, node.y, 1000);
      fgRef.current?.zoom(4, 1000);
    }
  }, []);

  const focusGraphNode = useCallback((fileName, folderName) => {
    if (!fgRef.current) return;
    const nodeId = folderName ? `${folderName}/${fileName}` : fileName;
    const node = data.nodes.find(n => n.id === nodeId || n.name === fileName);
    if (node) {
      fgRef.current.centerAt(node.x, node.y, 800);
      fgRef.current.zoom(5, 800);
      setActiveTab('browse');
    }
  }, [data.nodes]);

  // ─── Standout Feature Handlers ──────────────────────────────────

  const playAmbientSound = useCallback((type) => {
    if (!soundEnabled) return;
    switch (type) {
      case 'cluster': playClusterChime(); break;
      case 'move': playFileMoveSwish(); break;
      case 'upload': playUploadPing(); break;
      case 'response': playCommandResponse(); break;
      case 'duplicate': playDuplicateAlert(); break;
      default: break;
    }
  }, [soundEnabled]);

  const handleNlCommand = async () => {
    if (!nlCommand.trim()) return;
    setNlLoading(true);
    setNlResponse(null);
    try {
      const res = await axios.post(`${SEFS_API}/nl-command`, { command: nlCommand });
      setNlResponse(res.data);
      setNlCommand('');

      // Auto-refresh graph if files were moved/reorganized/deleted
      const movedCount = res.data.moved_files || 0;
      const deletedCount = res.data.deleted_files || 0;
      const didMutate = movedCount > 0 || deletedCount > 0 || res.data.action === 'move' || res.data.action === 'organize' || res.data.action === 'delete';
      if (didMutate) {
        playAmbientSound('move');
        // Short delay to let DB commits settle, then refresh
        setTimeout(() => fetchStructure(), 300);
        if (movedCount > 0) {
          showFeedback(`Moved ${movedCount} file(s) successfully`, 'success');
        }
        if (deletedCount > 0) {
          showFeedback(`Deleted ${deletedCount} file(s)`, 'success');
        }
      } else {
        playAmbientSound('response');
      }

      // Auto-show search results
      if (res.data.action === 'search' && res.data.search_results) {
        setSearchResults(res.data.search_results);
      }
    } catch (err) {
      showFeedback('Failed to process command', 'error');
    }
    setNlLoading(false);
  };

  const fetchTimeline = async () => {
    try {
      const res = await axios.get(`${SEFS_API}/cluster-history`);
      setTimelineSnapshots(res.data.snapshots);
      setTimelineIndex(0); // Most recent
    } catch (err) {
      console.error(err);
    }
  };

  const loadSnapshot = async (id) => {
    try {
      const res = await axios.get(`${SEFS_API}/cluster-snapshot/${id}`);
      // Reconstruct graph data from snapshot
      // This is a visualization-only state, doesn't affect actual backend
      const snapshot = res.data.snapshot;
      const nodes = [];
      const links = [];

      Object.entries(snapshot).forEach(([folder, members]) => {
        nodes.push({ id: folder, group: 'folder', name: folder, val: 20 });
        members.forEach(m => {
          const fileId = `${folder}/${m.file}`;
          nodes.push({ id: fileId, group: 'file', name: m.file, folder: folder, val: 5 });
          links.push({ source: folder, target: fileId });
        });
      });

      setData({ nodes, links });
    } catch (err) {
      showFeedback('Failed to load snapshot', 'error');
    }
  };

  const fetchDuplicates = async () => {
    try {
      const res = await axios.get(`${SEFS_API}/duplicates`);
      setDuplicates(res.data.duplicates);
      if (res.data.duplicates.length > 0) playAmbientSound('duplicate');
    } catch (err) {
      console.error(err);
    }
  };

  const fetchCrossEdges = async () => {
    try {
      const res = await axios.get(`${SEFS_API}/cross-edges`);
      setCrossEdges(res.data.edges);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchEntropy = async () => {
    try {
      const res = await axios.get(`${SEFS_API}/entropy`);
      setEntropyData(res.data.entropy);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchFolderSummary = async (folderName) => {
    if (folderSummaries[folderName]) return;
    try {
      const res = await axios.get(`${SEFS_API}/folder-summary/${folderName}`);
      setFolderSummaries(prev => ({ ...prev, [folderName]: res.data }));
    } catch (err) {
      console.error(err);
    }
  };

  const scanDirectory = async () => {
    if (!scanPath.trim()) {
      showFeedback('Please enter a directory path', 'error');
      return;
    }

    setLoadingFile(true);
    try {
      const res = await axios.post(`${SEFS_API}/scan-directory`, { path: scanPath });
      showFeedback(`Scanned ${res.data.files_processed} files, organized ${res.data.clusters.length} clusters`, 'success');
      setScanPath('');
      fetchStructure();
    } catch (err) {
      const msg = err.response?.data?.detail || 'Scan failed';
      showFeedback(msg, 'error');
    }
    setLoadingFile(false);
  };

  const generateReport = async () => {
    try {
      const res = await axios.get(`${SEFS_API}/export-report`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sefs_semantic_report.md');
      document.body.appendChild(link);
      link.click();
      link.remove();
      showFeedback('Report downloaded');
    } catch (err) {
      showFeedback('Failed to generate report', 'error');
    }
  };

  const performSearch = async (query) => {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    try {
      const res = await axios.get(`${SEFS_API}/search`, { params: { q: query, limit: 10 } });
      setSearchResults(res.data);
    } catch (err) {
      console.error('Search failed:', err);
      showFeedback('Search failed', 'error');
      setSearchResults(null);
    }
    setSearching(false);
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const formatUptime = (seconds) => {
    if (!seconds) return '00:00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const modelName = systemInfo?.llm_model || systemInfo?.model_name || 'llama-3.3-70b';
  const modelDisplayName = modelName.split('-').map(s => s.toUpperCase()).join('-');
  const storageRoot = systemInfo?.upload_storage || systemInfo?.monitored_root || 'D:/SEFS/upload_storage';
  const folders = fileStructure.filter(i => i.type === 'folder');

  // ─── Tab Content Renderers ─────────────────────────────────────────────

  const renderBrowseView = () => (
    <div ref={graphContainerRef} className="w-full h-full relative overflow-hidden">

      {/* ─── Unified Bottom Dock (NL Bar + Toggles) ─── */}
      <div className="absolute z-30 flex flex-col items-center gap-3 pointer-events-none" style={{ bottom: '20px', left: '50%', transform: 'translateX(-50%)', width: '92%', maxWidth: '1100px' }}>

        {/* Timeline Slider (appears above dock when active) */}
        <AnimatePresence>
          {showTimeline && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="pointer-events-auto glass-panel p-4 rounded-2xl flex flex-col gap-3"
              style={{ width: '100%', maxWidth: '520px' }}
            >
              <div className="flex justify-between items-center text-xs font-bold uppercase tracking-widest">
                <span className="flex items-center gap-2" style={{ color: 'var(--primary)' }}><History size={14} /> Semantic History</span>
                <span className="font-mono" style={{ color: 'var(--text-dim)' }}>{timelineSnapshots[timelineIndex]?.timestamp ? new Date(timelineSnapshots[timelineIndex].timestamp * 1000).toLocaleTimeString() : 'Current'}</span>
              </div>
              <input
                type="range"
                min="0"
                max={Math.max(0, timelineSnapshots.length - 1)}
                value={timelineIndex}
                onChange={e => {
                  const idx = parseInt(e.target.value);
                  setTimelineIndex(idx);
                  if (timelineSnapshots[idx]) loadSnapshot(timelineSnapshots[idx].id);
                }}
                className="w-full accent-indigo-500 h-1.5 rounded-lg appearance-none cursor-pointer"
                style={{ background: 'var(--input-border)' }}
              />
              <div className="flex justify-between text-[10px] font-mono" style={{ color: 'var(--text-dim)' }}>
                <span>Past</span>
                <span>Present</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* NL Response Bubble (appears above dock) */}
        <AnimatePresence>
          {nlResponse && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="pointer-events-auto glass-panel p-4 rounded-2xl w-full"
              style={{ borderLeft: '3px solid var(--primary)' }}
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg flex-shrink-0" style={{ background: 'rgba(99,102,241,0.1)' }}>
                  <Sparkles size={14} style={{ color: 'var(--primary)' }} />
                </div>
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  <p className="text-sm font-medium leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{nlResponse.explanation}</p>
                  {nlResponse.action === 'move' && nlResponse.move_evaluation && (
                    <div className="text-xs mt-2 p-2.5 rounded-xl flex items-center gap-4" style={{ background: 'var(--card-bg)', border: '1px solid var(--card-border)' }}>
                      <span style={{ color: 'var(--text-dim)' }}>Coherence Impact:</span>
                      <span className="font-bold" style={{ color: nlResponse.move_evaluation.impact > 0 ? '#22c55e' : '#ef4444' }}>
                        {nlResponse.move_evaluation.impact > 0 ? '+' : ''}{nlResponse.move_evaluation.impact} pts
                      </span>
                    </div>
                  )}
                </div>
                <button onClick={() => setNlResponse(null)} className="flex-shrink-0 p-1.5 rounded-lg transition-all hover:bg-white/5" style={{ color: 'var(--text-dim)' }}><X size={14} /></button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── Main Dock Bar ─── */}
        <div className="pointer-events-auto glass-panel flex items-center rounded-2xl w-full" style={{ padding: '6px 6px 6px 8px' }}>

          {/* Toggle Buttons — compact pills */}
          <div className="flex items-center gap-1 flex-shrink-0" style={{ marginRight: '12px', paddingRight: '12px', borderRight: '1px solid var(--glass-border)' }}>
            {[
              { active: showTimeline, onClick: () => { setShowTimeline(!showTimeline); if (!showTimeline) fetchTimeline(); }, icon: History, tip: 'Time Travel' },
              { active: showEntropy, onClick: () => { setShowEntropy(!showEntropy); if (!showEntropy) fetchEntropy(); }, icon: Thermometer, tip: 'Entropy Map' },
              { active: soundEnabled, onClick: () => { const ns = !soundEnabled; setSoundEnabled(ns); localStorage.setItem('sefs-sound', ns); }, icon: soundEnabled ? Volume2 : VolumeX, tip: soundEnabled ? 'Sound On' : 'Sound Off' }
            ].map((btn, i) => (
              <button
                key={i}
                onClick={btn.onClick}
                title={btn.tip}
                className="transition-all duration-200"
                style={{
                  padding: '8px 10px',
                  borderRadius: '10px',
                  display: 'flex', alignItems: 'center', gap: '6px',
                  background: btn.active ? 'linear-gradient(135deg, var(--primary), var(--secondary))' : 'transparent',
                  color: btn.active ? '#fff' : 'var(--text-dim)',
                  border: 'none', cursor: 'pointer',
                  boxShadow: btn.active ? '0 4px 12px rgba(99,102,241,0.3)' : 'none',
                  fontSize: '11px', fontWeight: 600, letterSpacing: '0.02em'
                }}
              >
                <btn.icon size={15} />
              </button>
            ))}
          </div>

          {/* AI Command Input */}
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="flex items-center gap-1.5 flex-shrink-0 px-2.5 py-1 rounded-lg" style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08))' }}>
              <Sparkles size={11} style={{ color: 'var(--primary)' }} />
              <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.08em', color: 'var(--primary)', textTransform: 'uppercase' }}>AI</span>
            </div>
            <input
              ref={nlInputRef}
              type="text"
              value={nlCommand}
              onChange={e => setNlCommand(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleNlCommand()}
              placeholder='Ask AI: "Move budget files to Finance", "Explain this cluster", "Find duplicates"...'
              disabled={nlLoading}
              style={{
                background: 'none', border: 'none', outline: 'none',
                color: 'var(--text-secondary)', fontSize: '13px', fontWeight: 500,
                fontFamily: 'inherit', letterSpacing: '0.01em', width: '100%',
                height: '36px', caretColor: 'var(--primary)',
                fontStyle: nlCommand ? 'normal' : 'normal'
              }}
            />
            {nlLoading && <Loader size={14} className="spin flex-shrink-0" style={{ color: 'var(--primary)' }} />}
            <button
              onClick={handleNlCommand}
              disabled={nlLoading || !nlCommand.trim()}
              className="flex-shrink-0 transition-all duration-200"
              style={{
                padding: '8px 14px',
                borderRadius: '10px',
                display: 'flex', alignItems: 'center', gap: '6px',
                background: nlCommand.trim() ? 'linear-gradient(135deg, var(--primary), var(--secondary))' : 'var(--card-bg)',
                color: nlCommand.trim() ? '#fff' : 'var(--text-dim)',
                border: nlCommand.trim() ? 'none' : '1px solid var(--card-border)',
                cursor: nlCommand.trim() ? 'pointer' : 'not-allowed',
                boxShadow: nlCommand.trim() ? '0 4px 12px rgba(99,102,241,0.25)' : 'none',
                fontSize: '12px', fontWeight: 600
              }}
            >
              <Send size={13} />
            </button>
          </div>
        </div>
      </div>

      {/* ─── Folder Summary Card (Top Right on hover) ─── */}
      <AnimatePresence>
        {hoveredCluster && folderSummaries[hoveredCluster] && (
          <motion.div
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="absolute z-30 p-5 rounded-xl pointer-events-none shadow-2xl glass-panel"
            style={{ top: '24px', right: '24px', maxWidth: '300px', minWidth: '200px', background: 'var(--bg-surface)', backdropFilter: 'blur(16px)', border: '1px solid var(--card-border-hover)', borderLeft: '4px solid var(--secondary)' }}
          >
            <h4 className="font-bold flex items-center gap-2 text-sm mb-2" style={{ color: 'var(--text-main)' }}>
              <Sparkles size={14} style={{ color: 'var(--secondary)' }} />
              {folderSummaries[hoveredCluster].suggested_name || hoveredCluster}
            </h4>
            <p className="text-xs leading-relaxed mb-3" style={{ color: 'var(--text-secondary)' }}>
              {folderSummaries[hoveredCluster].summary || 'Analyzing folder contents...'}
            </p>
            {folderSummaries[hoveredCluster].file_count != null && (
              <p className="text-xs mb-2" style={{ color: 'var(--text-dim)', fontSize: '10px' }}>
                {folderSummaries[hoveredCluster].file_count} file(s)
              </p>
            )}
            <div className="flex items-center gap-2 text-[10px] font-mono" style={{ color: 'var(--text-dim)' }}>
              <span style={{ fontSize: '9px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Coherence</span>
              <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--card-border)' }}>
                <div className="h-full rounded-full" style={{ width: `${folderSummaries[hoveredCluster].coherence_score * 100}%`, background: 'var(--secondary)' }} />
              </div>
              <span className="font-bold" style={{ color: 'var(--text-secondary)' }}>{Math.round(folderSummaries[hoveredCluster].coherence_score * 100)}%</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Entropy Legend (Top Left) ─── */}
      {showEntropy && (
        <div className="absolute top-6 left-6 pointer-events-none p-4 rounded-xl z-20 flex flex-col gap-2.5 glass-panel" style={{ background: 'var(--bg-surface)', backdropFilter: 'blur(16px)', border: '1px solid var(--card-border-hover)', boxShadow: '0 4px 16px rgba(0,0,0,0.15)' }}>
          <div className="flex items-center gap-2 text-[10px] uppercase font-bold tracking-widest pb-1" style={{ color: 'var(--text-dim)', borderBottom: '1px solid var(--card-border)' }}>
            <Thermometer size={12} /> File Entropy
          </div>
          <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}><div className="w-2.5 h-2.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"></div> Stable</div>
          <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}><div className="w-2.5 h-2.5 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.5)]"></div> Shifting</div>
          <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}><div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div> Volatile</div>
        </div>
      )}

      {data.nodes.length === 0 || data.nodes.length === 1 ? (
        <div className="flex flex-col items-center justify-center h-full gap-8 z-10 relative">
          <div className="p-8 rounded-full" style={{ background: 'rgba(99, 102, 241, 0.05)', border: '1px solid rgba(99, 102, 241, 0.1)' }}>
            <FolderOpen size={64} strokeWidth={1} className="text-indigo-400" style={{ opacity: 0.8 }} />
          </div>
          <div className="flex flex-col items-center gap-3">
            <h3 className="text-xl font-heading font-bold gradient-text">Ready for Files</h3>
            <p className="text-sm text-slate-500 text-center max-w-sm leading-relaxed">
              Upload documents, code, or archives. The Semantic Engine will auto-organize everything into intelligent clusters.
            </p>
          </div>
          <button
            className="action-btn-primary"
            onClick={() => setShowUpload(true)}
            style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '16px 32px', borderRadius: '16px', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff', border: 'none', fontSize: '15px', fontWeight: '600', cursor: 'pointer', boxShadow: '0 6px 20px rgba(99, 102, 241, 0.35)', zIndex: 10, position: 'relative' }}
          >
            <Upload size={18} /> Upload Files
          </button>
        </div>
      ) : graphDimensions.width > 0 && graphDimensions.height > 0 ? (
        <ForceGraph2D
          ref={fgRef}
          width={graphDimensions.width}
          height={graphDimensions.height}
          graphData={data}
          nodeLabel="name"
          nodeAutoColorBy="group"
          linkColor={() => {
            const resolved = theme === 'system' ? getSystemTheme() : theme;
            return resolved === 'light' ? '#6366f140' : '#ffffff4d';
          }}
          linkWidth={link => crossEdges.find(e => e.source === link.source.name && e.target === link.target.name) ? 2 : 1.8}
          linkLineDash={link => crossEdges.find(e => e.source === link.source.name && e.target === link.target.name) ? [4, 2] : null}
          backgroundColor="#00000000"
          onNodeClick={handleNodeClick}
          onNodeHover={node => {
            if (node && node.group === 'folder') {
              setHoveredCluster(node.name);
              fetchFolderSummary(node.name);
            } else {
              setHoveredCluster(null);
            }
          }}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTicks={100}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const resolved = theme === 'system' ? getSystemTheme() : theme;
            const isLight = resolved === 'light';
            const label = node.name;
            const fontSize = 12 / globalScale;
            ctx.font = `${fontSize}px Outfit`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Entropy colors
            let color;
            if (showEntropy && node.group === 'file' && entropyData[node.name]) {
              const s = entropyData[node.name].stability;
              color = s === 'stable' ? '#22c55e' : s === 'shifting' ? '#eab308' : '#ef4444';
            } else {
              const colors = { root: '#6366f1', folder: '#c084fc', file: isLight ? '#64748b' : '#94a3b8' };
              color = colors[node.group] || (isLight ? '#334155' : '#ffffff');
            }

            // Duplicate warning halo
            if (node.group === 'file' && duplicates.some(d => d.file_a === node.name || d.file_b === node.name)) {
              ctx.beginPath();
              ctx.arc(node.x, node.y, node.val + 8 / globalScale, 0, 2 * Math.PI, false);
              ctx.fillStyle = '#ef444440';
              ctx.fill();
              ctx.strokeStyle = '#ef4444';
              ctx.lineWidth = 1 / globalScale;
              ctx.stroke();
            }

            ctx.beginPath();
            ctx.arc(node.x, node.y, node.val + 2, 0, 2 * Math.PI, false);
            ctx.fillStyle = `${color}22`;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = isLight ? '#00000015' : '#ffffff22';
            ctx.lineWidth = 1 / globalScale;
            ctx.stroke();
            if (globalScale > 2 || node.group !== 'file') {
              ctx.fillStyle = isLight ? '#0f172a' : '#ffffff';
              ctx.font = `${fontSize * 1.1}px Space Grotesk`;
              ctx.fillText(label, node.x, node.y + (node.val * 1.5) + (5 / globalScale));
            }
          }}
        />
      ) : null}
    </div>
  );

  const renderLayersView = () => (
    <div className="flex flex-col h-full overflow-y-auto sidebar-scroll">
      <div className="p-6 max-w-7xl mx-auto w-full flex flex-col min-h-full">
        <div className="flex items-center justify-between mb-10">
          <div className="flex items-center gap-4">
            <div className="p-2.5 rounded-xl" style={{ background: 'rgba(168,85,247,0.1)' }}>
              <Layers size={18} className="text-purple-400" />
            </div>
            <div className="flex flex-col gap-1.5">
              <h2 className="font-heading font-semibold tracking-tight">Semantic Layers</h2>
              <span style={{ fontSize: '10px' }} className="text-slate-500">AI-organized folder structure · Click to expand</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="action-btn" onClick={() => setShowUpload(true)} title="Upload file">
              <Upload size={14} />
            </button>
            <button className="action-btn" onClick={fetchStructure} title="Refresh">
              <RefreshCw size={14} />
            </button>
          </div>
        </div>

        {/* Create folder */}
        <div className="flex items-center gap-2 mb-10">
          <input
            type="text"
            placeholder="New folder name..."
            value={newFolderName}
            onChange={e => setNewFolderName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && createFolder()}
            className="search-input flex-1 p-4 rounded-xl"
            style={{ background: 'var(--card-bg)', border: '1px solid var(--input-border)' }}
          />
          <button className="action-btn-primary" onClick={createFolder} disabled={creatingFolder || !newFolderName.trim()}>
            <Plus size={14} /> Create
          </button>
        </div>

        {folders.length === 0 ? (
          <div className="flex flex-col items-center justify-center flex-1 opacity-30 gap-6">
            <FolderOpen size={56} strokeWidth={1} />
            <p className="text-xs uppercase tracking-widest font-medium">No semantic folders yet</p>
            <p style={{ fontSize: '10px', maxWidth: '280px' }} className="text-slate-500 text-center">
              Upload files or drop them into the monitored root. The AI engine will auto-categorize them.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {folders.map((folder, i) => {
              const isExpanded = expandedFolder === folder.name;
              const fileDetails = folder.file_details || (folder.files || []).map(f =>
                typeof f === 'string' ? { name: f } : f
              );
              return (
                <motion.div
                  key={folder.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="glass-card rounded-2xl overflow-hidden p-2"
                >
                  {/* Folder Header */}
                  <div
                    className="p-6 flex items-center gap-4 cursor-pointer"
                    onClick={() => setExpandedFolder(isExpanded ? null : folder.name)}
                  >
                    <div className="p-2 rounded-xl bg-indigo-500/10 flex-shrink-0">
                      <Folder size={18} className="text-indigo-400" />
                    </div>
                    <span className="font-heading font-semibold text-sm flex-1 truncate">{folder.name}</span>
                    <span style={{ fontSize: '10px' }} className="font-bold px-2 py-1 rounded-lg bg-white/5 text-slate-500 tabular-nums">
                      {fileDetails.length} files
                    </span>
                    <button className="action-btn-sm" onClick={e => { e.stopPropagation(); deleteFolder(folder.name); }} title="Delete folder">
                      <Trash2 size={12} />
                    </button>
                    <ChevronDown size={16} className={`text-slate-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`} style={{ transition: 'transform 0.2s' }} />
                  </div>

                  {/* Expanded File List */}
                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="px-4 pb-4 flex flex-col gap-2" style={{ borderTop: '1px solid var(--input-border)' }}>
                          {fileDetails.length === 0 ? (
                            <p style={{ fontSize: '10px' }} className="text-slate-600 py-3 text-center">Folder is empty</p>
                          ) : fileDetails.map(file => {
                            const fname = typeof file === 'string' ? file : file.name;
                            return (
                              <div key={fname} className="flex items-center gap-2 py-2 px-3 rounded-lg group" style={{ background: 'var(--card-bg)' }}>
                                <FileText size={13} className="text-slate-600 flex-shrink-0" />
                                <span className="text-sm truncate flex-1">{fname}</span>
                                {file.size && <span style={{ fontSize: '9px' }} className="text-slate-600 flex-shrink-0">{formatBytes(file.size)}</span>}
                                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100" style={{ transition: 'opacity 0.2s' }}>
                                  <button className="action-btn-xs" onClick={() => viewFile(fname, folder.name)} title="View"><Eye size={18} /></button>
                                  <button className="action-btn-xs" onClick={() => downloadFile(fname, folder.name)} title="Download"><Download size={18} /></button>
                                  <button className="action-btn-xs" onClick={() => setShowMoveModal({ filename: fname, folder: folder.name })} title="Move"><FolderInput size={18} /></button>
                                  <button
                                    className="action-btn-xs"
                                    onClick={() => recategorizeFile(fname, folder.name)}
                                    title="Re-analyze with AI"
                                    disabled={recategorizing === `${folder.name}/${fname}`}
                                  >
                                    {recategorizing === `${folder.name}/${fname}` ? <RefreshCw size={18} className="spin" /> : <Cpu size={18} />}
                                  </button>
                                  <button className="action-btn-xs action-btn-danger" onClick={() => deleteFile(fname, folder.name)} title="Delete"><Trash2 size={18} /></button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );

  const renderDatabaseView = () => {
    const allFiles = [];
    fileStructure.forEach(item => {
      if (item.type === 'folder') {
        const fileDetails = item.file_details || (item.files || []).map(f => typeof f === 'string' ? { name: f } : f);
        fileDetails.forEach(f => allFiles.push({ ...f, name: f.name || f, folder: item.name }));
      } else {
        allFiles.push({ name: item.name, folder: 'Root', size: item.size, modified: item.modified });
      }
    });
    const filtered = searchQuery
      ? allFiles.filter(f => f.name?.toLowerCase().includes(searchQuery.toLowerCase()) || f.folder?.toLowerCase().includes(searchQuery.toLowerCase()))
      : allFiles;

    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex flex-col h-full p-6 max-w-7xl mx-auto w-full">
          <div className="flex items-center justify-between mb-10">
            <div className="flex items-center gap-4">
              <div className="p-2.5 rounded-xl" style={{ background: 'rgba(34,211,238,0.1)' }}>
                <Database size={18} style={{ color: '#22d3ee' }} />
              </div>
              <div className="flex flex-col gap-1.5">
                <h2 className="font-heading font-semibold tracking-tight">Knowledge Base</h2>
                <span style={{ fontSize: '10px' }} className="text-slate-500">{allFiles.length} indexed files · Click to view content</span>
              </div>
            </div>
            <button className="action-btn" onClick={() => setShowUpload(true)} title="Upload file">
              <Upload size={14} /> Upload
            </button>
          </div>

          <div className="search-bar flex items-center gap-4 mb-10 p-4 rounded-xl">
            <Search size={16} className="text-slate-500 flex-shrink-0" />
            <input
              type="text"
              placeholder="Search files or folders..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            {searchQuery && (
              <span style={{ fontSize: '10px' }} className="text-slate-500 flex-shrink-0">{filtered.length} results</span>
            )}
          </div>

          <div className="flex-1 overflow-y-auto sidebar-scroll pr-2">
            {filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 opacity-30 gap-4">
                <Search size={48} strokeWidth={1} />
                <p className="text-xs uppercase tracking-widest font-medium">
                  {searchQuery ? 'No matches found' : 'No files indexed'}
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                {filtered.map((file, i) => (
                  <motion.div
                    key={`${file.folder}/${file.name}-${i}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: Math.min(i * 0.02, 0.5) }}
                    className="glass-card p-6 rounded-xl flex items-center gap-4 group cursor-pointer"
                    onClick={() => viewFile(file.name, file.folder === 'Root' ? null : file.folder)}
                  >
                    <div className="p-2 rounded-lg bg-white/5 flex-shrink-0">
                      <FileText size={14} className="text-slate-400" />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="text-sm font-medium truncate">{file.name}</span>
                      <div className="flex items-center gap-3" style={{ fontSize: '10px' }}>
                        <span className="flex items-center gap-1">
                          <Folder size={9} className="text-purple-400" />
                          <span className="text-purple-400/80">{file.folder}</span>
                        </span>
                        {file.size && <span className="text-slate-600">{formatBytes(file.size)}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100" style={{ transition: 'opacity 0.2s' }}>
                      {file.folder !== 'Root' && (
                        <button className="action-btn-xs" onClick={e => { e.stopPropagation(); downloadFile(file.name, file.folder); }} title="Download">
                          <Download size={18} />
                        </button>
                      )}
                      <button className="action-btn-xs" onClick={e => { e.stopPropagation(); viewFile(file.name, file.folder === 'Root' ? null : file.folder); }} title="View">
                        <Eye size={20} className="text-indigo-400" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderStatsView = () => {
    const bs = backendStats || {};
    return (
      <div className="flex flex-col h-full overflow-y-auto sidebar-scroll">
        <div className="p-6 max-w-7xl mx-auto w-full flex flex-col min-h-full">
          <div className="flex items-center justify-between mb-10">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl" style={{ background: 'rgba(16,185,129,0.1)' }}>
                <BarChart3 size={18} style={{ color: '#34d399' }} />
              </div>
              <div className="flex flex-col gap-1.5">
                <h2 className="font-heading font-semibold tracking-tight">System Metrics</h2>
                <span style={{ fontSize: '10px' }} className="text-slate-500">Live telemetry · Auto-refreshes every 5s</span>
              </div>
            </div>
            <button className="action-btn" onClick={fetchBackendStats} title="Refresh now">
              <RefreshCw size={14} />
            </button>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            {[
              { label: 'Semantic Folders', value: bs.folders ?? '—', icon: Folder, color: '#818cf8' },
              { label: 'Files Indexed', value: bs.files ?? '—', icon: FileText, color: '#c084fc' },
              { label: 'Chunks Embedded', value: bs.chunks ?? 0, icon: Zap, color: '#22d3ee' },
              { label: 'Uptime', value: formatUptime(bs.uptime_seconds), icon: Clock, color: '#34d399' },
              { label: 'Total Size', value: formatBytes(bs.total_size_bytes), icon: HardDrive, color: '#f59e0b' },
              { label: 'API Calls', value: bs.api_calls ?? 0, icon: Server, color: '#818cf8' },
              { label: 'WS Connections', value: bs.websocket_connections ?? 0, icon: Share2, color: '#c084fc' },
              { label: 'Errors', value: bs.errors ?? 0, icon: AlertCircle, color: bs.errors > 0 ? '#ef4444' : '#34d399' }
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                className="glass-card p-6 rounded-2xl flex flex-col gap-2"
              >
                <div className="p-2 rounded-lg w-fit" style={{ background: `${stat.color}15` }}>
                  <stat.icon size={16} style={{ color: stat.color }} />
                </div>
                <span className="text-xl font-bold font-heading tabular-nums">{stat.value}</span>
                <span style={{ fontSize: '10px' }} className="text-slate-500 uppercase tracking-wider">{stat.label}</span>
              </motion.div>
            ))}
          </div>

          {/* Live status indicators */}
          <div className="glass-card p-5 rounded-2xl mb-4">
            <div className="flex items-center gap-3 mb-4">
              <Cpu size={16} className="text-indigo-400" />
              <span className="text-xs font-bold uppercase tracking-widest text-indigo-300">Engine Status</span>
            </div>
            <div className="flex flex-col gap-3">
              {[
                { label: 'AI Model', value: modelDisplayName, dot: true },
                { label: 'Mode', value: 'Upload-Driven', dot: true },
                { label: 'WebSocket', value: `${bs.websocket_connections ?? 0} connections`, dot: (bs.websocket_connections ?? 0) > 0 },
                { label: 'Last Activity', value: bs.last_activity ? new Date(bs.last_activity).toLocaleTimeString() : 'No activity yet' }
              ].map(row => (
                <div key={row.label} className="flex items-center justify-between">
                  <span style={{ fontSize: '10px' }} className="text-slate-500">{row.label}</span>
                  <div className="flex items-center gap-2">
                    {row.dot !== undefined && <div className={`status-dot-sm ${!row.dot ? 'status-dot-inactive' : ''}`} />}
                    <span className="text-xs font-medium" style={{ color: row.dot === false ? '#94a3b8' : '#e2e8f0' }}>{row.value}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSettingsView = () => (
    <div className="flex flex-col h-full overflow-y-auto sidebar-scroll">
      <div className="p-6 max-w-7xl mx-auto w-full flex flex-col min-h-full">
        <div className="flex items-center gap-3 mb-10">
          <div className="p-2.5 rounded-xl bg-indigo-500/10">
            <SettingsIcon size={18} className="text-indigo-400" />
          </div>
          <div className="flex flex-col gap-2">
            <h2 className="font-heading font-semibold tracking-tight">Configuration</h2>
            <span style={{ fontSize: '10px' }} className="text-slate-500">System settings · Read from backend</span>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 rounded-2xl border border-indigo-500/30 bg-indigo-500/5">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-indigo-500/20 text-indigo-400">
                <Folder size={20} />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-indigo-100 mb-1">Connect Local Folder</h3>
                <p className="text-xs text-indigo-300/70 mb-4 leading-relaxed">
                  Directly organize a folder on your computer <b>in-place</b>. SEFS will scan, cluster, and create semantic subfolders inside your directory without moving files to storage.
                </p>

                <div className="flex gap-2">
                  <input
                    type="text"
                    value={scanPath}
                    onChange={(e) => setScanPath(e.target.value)}
                    placeholder="e.g. C:\Users\Documents\MyProject"
                    className="flex-1 bg-black/20 border border-indigo-500/30 rounded-xl px-4 py-2 text-sm text-indigo-100 placeholder:text-indigo-500/50 focus:outline-none focus:border-indigo-400 transition-colors code-font"
                  />
                  <button
                    onClick={scanDirectory}
                    disabled={loadingFile}
                    className="px-5 py-2 bg-indigo-500 hover:bg-indigo-400 text-white font-medium text-xs rounded-xl transition-all shadow-lg shadow-indigo-500/20 flex items-center gap-2"
                  >
                    {loadingFile ? <RefreshCw size={14} className="spin" /> : <Play size={14} fill="currentColor" />}
                    Scan & Organize
                  </button>
                </div>
              </div>
            </div>
          </div>

          {[
            { label: 'Upload Storage', value: storageRoot, icon: HardDrive, color: '#818cf8', mono: true },
            { label: 'AI Model', value: modelDisplayName, icon: Cpu, color: '#c084fc' },
            { label: 'API Provider', value: 'Cerebras AI', icon: Server, color: '#22d3ee' },
            { label: 'API Keys', value: `${systemInfo?.api_key_count || '—'} configured`, icon: Key, color: '#34d399' },
            { label: 'System Status', value: systemInfo?.status === 'active' ? 'Active' : 'Offline', icon: Shield, color: '#34d399' },
            { label: 'Backend URL', value: SEFS_API, icon: Share2, color: '#818cf8', mono: true }
          ].map((item, i) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass-card p-6 rounded-2xl flex items-center gap-4"
            >
              <div className="p-2.5 rounded-xl bg-white/5 flex-shrink-0">
                <item.icon size={18} style={{ color: item.color }} />
              </div>
              <div className="flex flex-col min-w-0 flex-1">
                <span style={{ fontSize: '10px' }} className="text-slate-500 uppercase tracking-wider mb-1">{item.label}</span>
                <span className={`text-sm font-medium truncate ${item.mono ? 'font-mono text-indigo-300' : ''}`}>{item.value}</span>
              </div>
              {item.label === 'System Status' && <div className="status-dot-sm" />}
            </motion.div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="mt-8">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Appearance</h3>
          <div className="glass-card p-5 rounded-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-white/5">
                {theme === 'light' ? <Sun size={16} style={{ color: '#f59e0b' }} /> : theme === 'dark' ? <Moon size={16} style={{ color: '#818cf8' }} /> : <Monitor size={16} style={{ color: '#22d3ee' }} />}
              </div>
              <div className="flex flex-col">
                <span style={{ fontSize: '12px' }} className="font-medium">Theme</span>
                <span style={{ fontSize: '10px', color: 'var(--text-dim)' }}>{theme === 'system' ? `System (${getSystemTheme()})` : theme === 'light' ? 'Light Mode' : 'Dark Mode'}</span>
              </div>
            </div>
            <div className="flex gap-2" style={{ background: 'var(--input-bg)', padding: '4px', borderRadius: '14px', border: '1px solid var(--input-border)' }}>
              {[
                { id: 'light', icon: Sun, label: 'Light' },
                { id: 'dark', icon: Moon, label: 'Dark' },
                { id: 'system', icon: Monitor, label: 'System' },
              ].map(opt => (
                <button
                  key={opt.id}
                  onClick={() => { setTheme(opt.id); localStorage.setItem('sefs-theme', opt.id); }}
                  className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl text-xs font-semibold transition-all"
                  style={{
                    background: theme === opt.id ? 'var(--primary)' : 'transparent',
                    color: theme === opt.id ? '#fff' : 'var(--text-muted)',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  <opt.icon size={14} />
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-8">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Quick Actions</h3>
          <div className="flex flex-col gap-2">
            <button className="action-btn-full" onClick={() => { setActiveTab('layers'); setShowUpload(true); }}>
              <Upload size={16} /> Upload & Categorize File
            </button>
            <button className="action-btn-full" onClick={reclusterFiles}>
              <RefreshCw size={16} /> Re-cluster All Files
            </button>
            <button className="action-btn-full" onClick={fetchStructure}>
              <RefreshCw size={16} /> Refresh File Structure
            </button>
            <button className="action-btn-full" onClick={clearAll} style={{ color: '#ef4444', borderColor: 'rgba(239,68,68,0.2)' }}>
              <Trash2 size={16} /> Clear All & Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAnalysisView = () => (
    <div className="h-full overflow-y-auto sidebar-scroll p-6 max-w-7xl mx-auto w-full">
      <div className="flex items-center justify-between mb-10">
        <div className="flex items-center gap-4">
          <div className="p-2.5 rounded-xl bg-orange-500/10 text-orange-400">
            <Activity size={18} />
          </div>
          <div className="flex flex-col gap-1.5">
            <h2 className="font-heading font-semibold tracking-tight">Semantic Analysis</h2>
            <span style={{ fontSize: '10px' }} className="text-slate-500">Duplicate detection & Cross-cluster relationships</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="action-btn" onClick={fetchDuplicates}>
            <RefreshCw size={14} /> Scan Duplicates
          </button>
          <button className="action-btn" onClick={fetchCrossEdges}>
            <Share2 size={14} /> Scan Relations
          </button>
          <button className="action-btn" onClick={generateReport}>
            <FileDown size={14} /> Export Report
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Duplicates Section */}
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
            <Copy size={14} /> Potential Duplicates
          </h3>
          {duplicates.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-xs italic border border-dashed border-slate-700 rounded-xl">
              No duplicates detected. Run a scan to check.
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {duplicates.map((d, i) => (
                <div key={i} className="p-3 bg-white/5 rounded-xl border border-white/5 flex flex-col gap-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-bold text-red-400">{(d.similarity * 100).toFixed(1)}% Match</span>
                    <span className="text-slate-500">Semantic Warning</span>
                  </div>
                  <div className="flex flex-col gap-1 text-sm text-slate-300">
                    <div className="truncate">{d.file_a}</div>
                    <div className="text-slate-500 text-[10px] text-center">vs</div>
                    <div className="truncate">{d.file_b}</div>
                  </div>
                  <button className="action-btn-xs w-full mt-1 bg-red-500/10 text-red-400 hover:bg-red-500/20" onClick={() => {
                    if (window.confirm(`Delete ${d.file_b}?`)) deleteFile(d.file_b, d.folder_b);
                  }}>
                    Remove Duplicate
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Cross Edges Section */}
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
            <GitBranch size={14} /> Cross-Cluster Links
          </h3>
          {crossEdges.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-xs italic border border-dashed border-slate-700 rounded-xl">
              No cross-cluster relationships found.
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {crossEdges.map((e, i) => (
                <div key={i} className="p-3 bg-white/5 rounded-xl border border-white/5 flex items-center gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-indigo-300 font-bold truncate">{e.source}</div>
                    <div className="text-[10px] text-slate-500">Source Cluster</div>
                  </div>
                  <ArrowRight size={12} className="text-slate-600" />
                  <div className="flex-1 min-w-0 text-right">
                    <div className="text-xs text-purple-300 font-bold truncate">{e.target}</div>
                    <div className="text-[10px] text-slate-500">Target Cluster</div>
                  </div>
                  <div className="text-[9px] font-mono bg-white/10 px-1.5 py-0.5 rounded text-slate-300">
                    {(e.weight * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'browse': return renderBrowseView();
      case 'layers': return renderLayersView();
      case 'database': return renderDatabaseView();
      case 'analysis': return renderAnalysisView();
      case 'stats': return renderStatsView();
      case 'settings': return renderSettingsView();
      default: return renderBrowseView();
    }
  };

  const getTabLabel = () => {
    const labels = {
      browse: 'Entropic Evolution Map',
      layers: 'Semantic Layer Manager',
      database: 'Knowledge Base Explorer',
      analysis: 'Semantic Analysis & Insights',
      stats: 'System Telemetry',
      settings: 'System Configuration'
    };
    return labels[activeTab] || 'Dashboard';
  };

  // ─── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen w-screen relative overflow-hidden text-slate-200">
      <div className="mesh-bg" />

      {/* ─── Sidebar ────────────────────────────────────────────────────── */}
      <aside className="fixed left-0 top-0 bottom-0 w-20 flex flex-col items-center py-8 z-30 border-r border-white/5 bg-slate-950/20 backdrop-blur-3xl">
        <div
          className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center p-2 neon-glow mb-12 cursor-pointer transition-transform hover:scale-110"
          onClick={() => setActiveTab('browse')}
          title="Dashboard Home"
        >
          <Share2 size={24} className="text-white" />
        </div>

        <nav className="flex flex-col gap-10 flex-1 w-full items-center">
          {[
            { id: 'browse', icon: Box, label: 'File Explorer' },
            { id: 'layers', icon: Layers, label: 'Semantic Layers' },
            { id: 'database', icon: Database, label: 'Knowledge Base' },
            { id: 'analysis', icon: Activity, label: 'Semantic Analysis' },
            { id: 'stats', icon: BarChart3, label: 'System Metrics' }
          ].map((item) => (
            <div
              key={item.id}
              className={`cursor-pointer transition-all duration-300 flex items-center justify-center w-full py-2 hover:scale-110 ${activeTab === item.id ? 'active-nav-link text-indigo-400 opacity-100' : 'text-slate-400 opacity-60 hover:opacity-100'}`}
              onClick={() => setActiveTab(item.id)}
              title={item.label}
            >
              <item.icon size={22} className="pointer-events-none" />
            </div>
          ))}

          <div className="mt-auto w-full flex flex-col items-center gap-6">
            {/* Quick theme toggle */}
            <div
              className="cursor-pointer transition-all duration-300 flex items-center justify-center w-full py-2 hover:scale-110 text-slate-400 opacity-60 hover:opacity-100"
              onClick={() => {
                const next = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark';
                setTheme(next);
                localStorage.setItem('sefs-theme', next);
              }}
              title={`Theme: ${theme === 'system' ? `System (${getSystemTheme()})` : theme} — Click to change`}
            >
              {theme === 'light' ? <Sun size={22} style={{ color: '#f59e0b' }} className="pointer-events-none" /> : theme === 'dark' ? <Moon size={22} style={{ color: '#818cf8' }} className="pointer-events-none" /> : <Monitor size={22} style={{ color: '#22d3ee' }} className="pointer-events-none" />}
            </div>
            <div
              className={`cursor-pointer transition-all duration-300 flex items-center justify-center w-full hover:scale-110 ${activeTab === 'settings' ? 'active-nav-link text-indigo-400 opacity-100' : 'text-slate-400 opacity-60 hover:opacity-100'}`}
              onClick={() => setActiveTab('settings')}
              title="System Settings"
            >
              <SettingsIcon size={22} className="pointer-events-none" />
            </div>
          </div>
        </nav>
      </aside>

      {/* ─── Main Content ───────────────────────────────────────────────── */}
      <div className="flex-1 ml-20 flex flex-col relative h-full">

        <header className="h-20 flex items-center justify-between px-10 z-20 mt-4">
          <h1 className="text-2xl font-bold tracking-tight pointer-events-none font-heading">
            <span className="gradient-text">SEFS</span>
            <span style={{ fontSize: '12px', letterSpacing: '0.3em' }} className="ml-3 opacity-40 font-normal uppercase font-sans">Semantic Intelligence</span>
          </h1>

          {/* Model & Storage Badge */}
          <div className="flex items-center gap-6 header-badge rounded-2xl px-5 py-3">
            <div className="flex items-center gap-3">
              <div className="model-indicator">
                <div className="model-indicator-ring" />
                <Cpu size={14} className="model-indicator-icon" />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="model-name-text font-bold" style={{ fontSize: '13px' }}>{modelDisplayName}</span>
                <span className="uppercase tracking-widest flex items-center gap-1.5" style={{ fontSize: '10px', color: 'rgba(52,211,153,0.8)' }}>
                  <Zap size={9} />
                  Cerebras · Active
                </span>
              </div>
            </div>

            <div className="h-8 w-px bg-white/10" />

            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5">
                <Database size={14} className="text-indigo-400" />
              </div>
              <div className="flex flex-col">
                <span style={{ fontSize: '10px' }} className="text-slate-500 uppercase font-bold tracking-tighter">Upload Storage</span>
                <code style={{ fontSize: '11px' }} className="text-indigo-300 font-mono tracking-tight">{storageRoot}</code>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 flex px-10 pb-8 gap-8 overflow-hidden h-full">
          <section className="flex-1 flex flex-col gap-6 relative h-full min-w-0">
            <div className="grid grid-cols-3 gap-6">
              {[
                { label: 'Semantic Folders', val: stats.folders, icon: Folder, color: 'text-indigo-400' },
                { label: 'Files Analyzed', val: stats.files, icon: FileText, color: 'text-purple-400' },
                { label: 'Realtime Ops', val: stats.ops, icon: Cpu, color: 'text-accent' }
              ].map((stat, i) => (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  key={i}
                  className="glass-panel p-5 rounded-2xl flex items-center justify-between min-w-0"
                >
                  <div className="flex flex-col">
                    <span className="text-sm text-slate-400 uppercase font-medium tracking-wider mb-1 truncate">{stat.label}</span>
                    <span className="text-3xl font-bold font-heading tabular-nums">{stat.val}</span>
                  </div>
                  <div className={`p-3 rounded-xl bg-white/5 ${stat.color} flex-shrink-0`}>
                    <stat.icon size={20} />
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Action Bar */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <button
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-all hover:scale-105"
                style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff' }}
                onClick={() => setShowUpload(true)}
              >
                <Upload size={15} /> Upload Files
              </button>
              <button
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-all hover:scale-105"
                style={{ background: 'rgba(99,102,241,0.1)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.2)' }}
                onClick={reclusterFiles}
              >
                <RefreshCw size={15} /> Re-cluster
              </button>

              {/* Search Bar — inline in action bar */}
              <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl flex-1" style={{ background: 'var(--card-bg)', border: '1px solid var(--input-border)' }}>
                <Search size={16} style={{ color: '#6366f1', flexShrink: 0 }} />
                <input
                  type="text"
                  placeholder="Semantic search across files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') performSearch(searchQuery); }}
                  style={{ background: 'none', border: 'none', outline: 'none', color: 'var(--text-secondary)', fontSize: '13px', width: '100%', fontFamily: 'inherit', letterSpacing: '0.01em' }}
                />
                {searchQuery && (
                  <button onClick={() => { setSearchQuery(''); setSearchResults(null); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', padding: 0, flexShrink: 0 }}>
                    <X size={13} />
                  </button>
                )}
                {searching ? (
                  <RefreshCw size={13} className="spin" style={{ color: '#818cf8', flexShrink: 0 }} />
                ) : searchQuery.trim() ? (
                  <button
                    onClick={() => performSearch(searchQuery)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, flexShrink: 0 }}
                  >
                    <ArrowRight size={13} style={{ color: '#6366f1' }} />
                  </button>
                ) : null}
              </div>

              <button
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-all hover:scale-105"
                style={{ background: 'rgba(239,68,68,0.08)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.15)', flexShrink: 0 }}
                onClick={clearAll}
              >
                <Trash2 size={14} /> Clear All
              </button>
            </div>

            <div className="flex-1 glass-panel rounded-3xl overflow-hidden relative border-white/5 min-h-0">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="h-full w-full"
                >
                  {renderTabContent()}
                </motion.div>
              </AnimatePresence>
            </div>
          </section>

          {/* Resizable Divider */}
          <div
            onMouseDown={onResizeStart}
            className="flex-shrink-0 flex items-center justify-center group"
            style={{ width: '12px', cursor: 'col-resize', zIndex: 10, margin: '0 -2px' }}
            title="Drag to resize"
          >
            <div style={{ width: '3px', height: '48px', borderRadius: '4px', background: 'var(--glass-border)', transition: 'background 0.2s, height 0.2s' }} className="group-hover:!bg-indigo-500" />
          </div>

          {/* Right Sidebar — resizable */}
          <aside className="flex flex-col gap-6 h-full flex-shrink-0" style={{ width: `${sidebarWidth}px` }}>
            <div className="glass-panel flex-1 rounded-3xl p-7 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-indigo-500/10 text-indigo-400">
                    {searchResults ? <Search size={18} /> : <Terminal size={18} />}
                  </div>
                  <h2 className="font-heading font-semibold tracking-tight">
                    {searchResults ? 'Search Results' : 'System Logs'}
                  </h2>
                </div>
                {searchResults ? (
                  <button
                    onClick={() => { setSearchResults(null); setSearchQuery(''); }}
                    className="text-[13px] font-semibold px-3 py-1.5 rounded-lg"
                    style={{ background: 'rgba(99,102,241,0.1)', color: '#818cf8', border: 'none', cursor: 'pointer' }}
                  >
                    ← Back to Logs
                  </button>
                ) : (
                  <span className="live-badge" style={{ fontSize: '11px', fontWeight: 700, padding: '4px 12px', borderRadius: '999px', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Live</span>
                )}
              </div>

              <div className="flex-1 overflow-y-auto sidebar-scroll pr-2">
                <div className="flex flex-col gap-4">
                  {searchResults ? (
                    /* Search Results View */
                    searchResults.results.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-24 opacity-20 gap-3">
                        <Search size={48} strokeWidth={1} />
                        <p className="text-sm uppercase tracking-widest font-medium">No Results Found</p>
                      </div>
                    ) : (
                      <>
                        <div className="text-sm text-slate-500 mb-2">
                          {searchResults.count} result{searchResults.count !== 1 ? 's' : ''} for "{searchResults.query}"
                        </div>
                        {searchResults.results.map((result, i) => (
                          <motion.div
                            key={`${result.file}-${i}`}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className="glass-card p-4 rounded-2xl flex flex-col gap-3 group cursor-pointer hover:bg-white/5 transition-colors"
                            onClick={() => {
                              focusGraphNode(result.file, result.folder);
                              viewFile(result.file, result.folder);
                            }}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full" style={{ background: `hsl(${Math.round(result.score * 120)}, 70%, 50%)` }} />
                                <span style={{ fontSize: '11px' }} className="font-bold text-indigo-400/80 uppercase tracking-widest">
                                  {Math.round(result.score * 100)}% Match
                                </span>
                              </div>
                              <span style={{ fontSize: '10px' }} className="opacity-30">#{i + 1}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="p-2 rounded-lg bg-white/5 flex-shrink-0">
                                <FileText size={14} className="text-slate-400" />
                              </div>
                              <div className="flex flex-col min-w-0 flex-1">
                                <span className="text-sm font-medium truncate">{result.file}</span>
                                {result.folder && (
                                  <div className="flex items-center gap-1.5" style={{ fontSize: '11px' }}>
                                    <Folder size={10} className="text-purple-400" />
                                    <span className="text-purple-400 font-semibold truncate">{result.folder}</span>
                                  </div>
                                )}
                              </div>
                              <ChevronRight size={14} className="text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                            </div>
                            {result.snippet && (
                              <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2" style={{ display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                                {result.snippet}
                              </p>
                            )}
                          </motion.div>
                        ))}
                      </>
                    )
                  ) : (
                    /* System Logs View */
                    <AnimatePresence initial={false}>
                      {logs.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-24 opacity-20 gap-3">
                          <Activity size={48} strokeWidth={1} />
                          <p className="text-sm uppercase tracking-widest font-medium">Idle - Waiting for I/O</p>
                        </div>
                      ) : logs.map((log, i) => (
                        <motion.div
                          key={`${i}-${log.file}-${log.timestamp}`}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="glass-card p-4 rounded-2xl flex flex-col gap-3 group"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                              <span style={{ fontSize: '11px' }} className="font-bold text-indigo-400/80 uppercase tracking-widest">
                                {log.type === 'delete' ? 'Deleted' :
                                  log.type === 'move' ? 'Semantic Move' :
                                    log.type === 'summary' ? 'Refined' :
                                      log.type === 'system' ? 'System' :
                                        log.type === 'upload' ? 'Ingested' :
                                          log.manual ? 'Manual Move' : 'Activity'}
                              </span>
                            </div>
                            <span style={{ fontSize: '10px' }} className="opacity-30">
                              {log.timestamp ? new Date(log.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'Just Now'}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-white/5 flex-shrink-0">
                              <FileText size={14} className="text-slate-400" />
                            </div>
                            <div className="flex flex-col min-w-0 flex-1">
                              <span className="text-sm font-medium truncate">{log.file || log.message || 'System Event'}</span>
                              {log.folder ? (
                                <div className="flex items-center gap-1.5" style={{ fontSize: '11px' }}>
                                  <ArrowRight size={10} className="text-indigo-400" />
                                  <span className="text-purple-400 font-semibold truncate">{log.folder}</span>
                                </div>
                              ) : log.file && (
                                <span className="text-[11px] text-slate-500 truncate">Event Processed</span>
                              )}
                            </div>
                            {/* ChevronRight removed based on user feedback */}
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  )}
                </div>
              </div>

              <div className="mt-8 p-4 glass-card rounded-2xl" style={{ background: 'rgba(99,102,241,0.05)', borderColor: 'rgba(99,102,241,0.08)' }}>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400"><Cpu size={14} /></div>
                  <span style={{ fontSize: '11px' }} className="font-bold uppercase tracking-widest text-indigo-300">LLM Engine Status</span>
                </div>
                <p style={{ fontSize: '11px', lineHeight: '1.7' }} className="text-slate-400 opacity-80">
                  Processing file entropy using <span className="model-name-inline font-bold">{modelDisplayName}</span> on Cerebras AI ultra-low latency infrastructure.
                </p>
              </div>
            </div>
          </aside >
        </main >
      </div >

      {
        createPortal(
          <>
            {/* ─── Action Feedback Toast ──────────────────────────────────────── */}
            < AnimatePresence >
              {actionFeedback && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="fixed top-6 left-1/2 z-100 toast-notification"
                  style={{ transform: 'translateX(-50%)' }}
                >
                  {actionFeedback.type === 'error' ? <AlertCircle size={14} style={{ color: '#ef4444' }} /> : <CheckCircle size={14} style={{ color: '#34d399' }} />}
                  <span>{actionFeedback.message}</span>
                </motion.div>
              )}
            </AnimatePresence >

            {/* ─── File Viewer Modal ──────────────────────────────────────────── */}
            {
              selectedFile && (
                <div
                  className="fixed inset-0 z-200 flex items-center justify-center"
                  style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(8px)' }}
                  onClick={closeFileViewer}
                >
                  <div
                    className="glass-panel rounded-[32px] p-10 w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden"
                    style={{ margin: '0 3rem', background: 'var(--bg-surface)', border: '1px solid var(--card-border-hover)', boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.7)' }}
                    onClick={e => e.stopPropagation()}
                  >
                    <div className="flex items-center justify-between mb-8" style={{ paddingBottom: '24px', borderBottom: '1px solid var(--input-border)' }}>
                      <div className="flex items-center gap-5">
                        <div className="p-4 rounded-2xl bg-indigo-500/15 text-indigo-400">
                          <FileText size={28} />
                        </div>
                        <div className="flex flex-col gap-1.5">
                          <span className="text-2xl font-heading font-bold gradient-text">{selectedFile.name}</span>
                          {selectedFile.folder && (
                            <span className="flex items-center gap-2" style={{ fontSize: '12px', color: '#c084fc', opacity: 0.8, fontWeight: 600 }}>
                              <Layers size={11} /> in {selectedFile.folder}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        {fileContent && !fileContent.error && selectedFile.folder && (
                          <>
                            <button className="action-btn" onClick={() => downloadFile(selectedFile.name, selectedFile.folder)} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 24px', borderRadius: '14px', background: 'var(--card-border)', border: '1px solid var(--card-border-hover)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px', fontWeight: 600 }}>
                              <Download size={16} /> Download
                            </button>
                            <button className="action-btn" onClick={() => setShowMoveModal({ filename: selectedFile.name, folder: selectedFile.folder })} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 24px', borderRadius: '14px', background: 'var(--card-border)', border: '1px solid var(--card-border-hover)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px', fontWeight: 600 }}>
                              <FolderInput size={16} /> Move
                            </button>
                            <button className="action-btn action-btn-danger" onClick={() => deleteFile(selectedFile.name, selectedFile.folder)} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', borderRadius: '14px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', cursor: 'pointer' }}>
                              <Trash2 size={18} />
                            </button>
                          </>
                        )}
                        <button className="action-btn" onClick={closeFileViewer} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', borderRadius: '14px', background: 'var(--card-border)', border: '1px solid var(--card-border-hover)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                          <X size={18} />
                        </button>
                      </div>
                    </div>

                    {loadingFile ? (
                      <div className="flex flex-col items-center justify-center py-32 gap-5" style={{ opacity: 0.4 }}>
                        <RefreshCw size={40} className="spin text-indigo-400" />
                        <span className="text-sm uppercase tracking-widest font-bold">Retrieving Semantic Content</span>
                      </div>
                    ) : fileContent?.error ? (
                      <div className="flex flex-col items-center justify-center py-32 gap-5" style={{ color: '#f87171' }}>
                        <AlertCircle size={40} />
                        <span className="font-semibold text-xl">{fileContent.error}</span>
                      </div>
                    ) : fileContent ? (
                      <div className="flex flex-col flex-1 overflow-hidden" style={{ gap: '1rem' }}>
                        <div className="flex items-center justify-between" style={{ paddingBottom: '16px', borderBottom: '1px solid var(--input-border)' }}>
                          <div className="flex items-center gap-8 text-gray-500" style={{ fontSize: '12px' }}>
                            <span className="flex items-center gap-2.5">Size: <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{formatBytes(fileContent.size)}</span></span>
                            <span className="flex items-center gap-2.5">Type: <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{fileContent.extension || 'unknown'}</span></span>
                            <span className="flex items-center gap-2.5">Modified: <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{new Date(fileContent.modified).toLocaleString()}</span></span>
                          </div>

                          {/* Toggle / Button Area */}
                          <div className="flex items-center gap-4">
                            {!summaryRequested ? (
                              <button
                                className="action-btn"
                                onClick={handleRequestSummary}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '8px',
                                  padding: '8px 16px',
                                  borderRadius: '10px',
                                  background: 'rgba(99, 102, 241, 0.1)',
                                  border: '1px solid rgba(99, 102, 241, 0.2)',
                                  color: '#818cf8',
                                  cursor: 'pointer',
                                  fontSize: '12px',
                                  fontWeight: 600
                                }}
                              >
                                <Cpu size={14} /> Get AI Summary
                              </button>
                            ) : (
                              <div className="flex items-center gap-4 bg-white/5 p-1 rounded-xl">
                                {showSummary && (
                                  <motion.button
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                    onClick={handleReadSummary}
                                    className={`p-2 rounded-lg flex items-center gap-2 text-xs font-bold transition-colors ${audioPlaying
                                      ? 'bg-red-500/20 text-red-400'
                                      : audioLoading
                                        ? 'bg-indigo-500/20 text-indigo-400 cursor-wait'
                                        : 'hover:bg-indigo-500/20 hover:text-indigo-400 text-slate-400'
                                      }`}
                                    title="Read Summary Aloud"
                                  >
                                    {audioLoading ? (
                                      <Loader size={16} className="animate-spin" />
                                    ) : audioPlaying ? (
                                      <>
                                        <Square size={16} fill="currentColor" />
                                        <span className="mr-1">STOP</span>
                                      </>
                                    ) : (
                                      <>
                                        <Volume2 size={16} />
                                        <span className="mr-1">READ</span>
                                      </>
                                    )}
                                  </motion.button>
                                )}

                                <div className="h-4 w-px bg-white/10" />

                                <button
                                  onClick={() => setShowSummary(false)}
                                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${!showSummary ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
                                >
                                  File Content
                                </button>
                                <button
                                  onClick={() => setShowSummary(true)}
                                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-2 ${showSummary ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
                                >
                                  <Cpu size={12} /> AI Summary
                                </button>
                              </div>
                            )}
                          </div>
                        </div>


                        {/* Content Area */}
                        <div className="flex-1 min-h-0 relative">
                          <AnimatePresence mode="wait">
                            {showSummary ? (
                              <motion.div
                                key="summary"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.2 }}
                                className="h-full overflow-y-scroll sidebar-scroll p-1"
                                style={{ height: '100%' }}
                              >
                                <div
                                  className="p-8 rounded-[24px] min-h-full"
                                  style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.15)', boxShadow: 'inset 0 0 30px rgba(99,102,241,0.05)' }}
                                >
                                  <div className="flex items-center gap-2 mb-4">
                                    <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-300">
                                      <Cpu size={14} />
                                    </div>
                                    <span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#818cf8' }}>AI Semantic Summary</span>
                                    {loadingSummary && <RefreshCw size={12} className="spin text-indigo-400 ml-2" />}
                                  </div>

                                  {loadingSummary ? (
                                    <div className="flex flex-col gap-4">
                                      <div className="h-4 w-3/4 bg-indigo-500/20 rounded animate-pulse" />
                                      <div className="h-4 w-full bg-indigo-500/10 rounded animate-pulse" />
                                      <div className="h-4 w-5/6 bg-indigo-500/10 rounded animate-pulse" />
                                      <div className="flex items-center gap-4 text-slate-500 mt-4" style={{ fontSize: '14px', fontStyle: 'italic' }}>
                                        <span className="animate-pulse">Synthesizing file insights...</span>
                                      </div>
                                    </div>
                                  ) : fileSummary ? (
                                    <p className="text-slate-200" style={{ fontSize: '14px', lineHeight: '1.8', letterSpacing: '0.015em', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                                      {fileSummary}
                                    </p>
                                  ) : (
                                    <p className="text-slate-500 italic">No summary available.</p>
                                  )}
                                </div>
                              </motion.div>
                            ) : (
                              <motion.div
                                key="content"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                transition={{ duration: 0.2 }}
                                className="h-full flex flex-col"
                              >
                                {fileContent.is_text && fileContent.content ? (
                                  <pre className="flex-1 overflow-auto p-6 rounded-[24px] font-mono text-sm sidebar-scroll" style={{ background: 'var(--code-bg)', border: '1px solid var(--card-border)', lineHeight: '1.8', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text-secondary)', maxHeight: '60vh' }}>
                                    {fileContent.content}
                                  </pre>
                                ) : fileContent.extension === '.pdf' ? (
                                  <div className="flex-1 flex flex-col gap-3 overflow-hidden" style={{ minHeight: '500px' }}>
                                    <iframe
                                      src={`${SEFS_API}/preview?filename=${encodeURIComponent(selectedFile.name)}${selectedFile.folder ? `&folder=${encodeURIComponent(selectedFile.folder)}` : ''}`}
                                      title={selectedFile.name}
                                      style={{
                                        width: '100%',
                                        flex: 1,
                                        borderRadius: '20px',
                                        border: '1px solid rgba(255,255,255,0.08)',
                                        background: '#1e1e2e',
                                        minHeight: '480px',
                                      }}
                                    />
                                  </div>
                                ) : (
                                  <div className="flex flex-col items-center justify-center h-full gap-5 bg-black/30 rounded-[28px] border border-white/5" style={{ opacity: 0.4 }}>
                                    <FileText size={64} strokeWidth={1} />
                                    <span className="text-base font-medium tracking-wide">Preview unavailable for binary formats</span>
                                    <button className="action-btn" onClick={() => downloadFile(selectedFile.name, selectedFile.folder)} style={{ marginTop: '12px', padding: '10px 24px', borderRadius: '12px', background: 'var(--card-bg)', border: '1px solid var(--card-border-hover)', color: 'var(--text-main)', cursor: 'pointer' }}>Download to View</button>
                                  </div>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              )
            }

            {/* ─── Move Modal ─────────────────────────────────────────────────── */}
            {
              showMoveModal && (
                <div
                  className="fixed inset-0 z-[300] flex items-center justify-center"
                  style={{ background: 'var(--overlay-bg)', backdropFilter: 'blur(8px)' }}
                  onClick={() => setShowMoveModal(null)}
                >
                  <div
                    className="glass-panel rounded-3xl p-6 w-full max-w-md"
                    style={{ margin: '0 2rem', background: 'var(--bg-surface)', border: '1px solid var(--card-border-hover)' }}
                    onClick={e => e.stopPropagation()}
                  >
                    <h3 className="font-heading font-semibold mb-1" style={{ fontSize: '18px' }}>Move File</h3>
                    <p style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '20px' }}>
                      Moving "{showMoveModal.filename}" from {showMoveModal.folder || 'Root'}
                    </p>
                    <div className="flex flex-col gap-2">
                      {folders.filter(f => f.name !== showMoveModal.folder).map(f => (
                        <button
                          key={f.name}
                          className="glass-card p-3 rounded-xl flex items-center gap-3 cursor-pointer text-left"
                          style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '12px', background: 'var(--card-bg)', border: '1px solid var(--input-border)', cursor: 'pointer' }}
                          onClick={() => moveFile(showMoveModal.filename, showMoveModal.folder, f.name)}
                        >
                          <Folder size={16} className="text-indigo-400" />
                          <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{f.name}</span>
                          <span style={{ fontSize: '10px', color: 'var(--text-dim)', marginLeft: 'auto' }}>{f.file_count ?? f.files?.length ?? 0} files</span>
                        </button>
                      ))}
                    </div>
                    <button
                      className="action-btn w-full mt-4 justify-center"
                      onClick={() => setShowMoveModal(null)}
                      style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '12px 24px', borderRadius: '12px', background: 'var(--card-border)', border: '1px solid var(--card-border-hover)', color: 'var(--text-secondary)', cursor: 'pointer', marginTop: '16px' }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )
            }

            {/* ─── Upload Modal ───────────────────────────────────────────────── */}
            {console.log('Rendering portal, showUpload:', showUpload)}
            {
              showUpload && (
                <div
                  className="fixed inset-0 flex items-center justify-center"
                  style={{
                    background: 'rgba(0,0,0,0.9)',
                    backdropFilter: 'blur(8px)',
                    zIndex: 9999,
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0
                  }}
                  onClick={() => {
                    console.log('Modal backdrop clicked');
                    setShowUpload(false);
                    setUploadResult(null);
                  }}
                >
                  <div
                    className="glass-panel rounded-3xl p-8 w-full max-w-md text-center relative"
                    style={{
                      margin: '0 2rem',
                      background: 'var(--bg-surface)',
                      border: '2px solid #6366f1',
                      zIndex: 10000,
                      position: 'relative'
                    }}
                    onClick={e => {
                      console.log('Modal content clicked');
                      e.stopPropagation();
                    }}
                  >
                    <button
                      onClick={() => {
                        console.log('Close button clicked');
                        setShowUpload(false);
                        setUploadResult(null);
                      }}
                      style={{ position: 'absolute', top: '16px', right: '16px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', zIndex: 10001 }}
                    >
                      <X size={20} />
                    </button>

                    <FolderInput size={32} className="text-indigo-400 mx-auto mb-4" />
                    <h3 className="font-heading font-semibold mb-2" style={{ fontSize: '18px', color: 'var(--text-main)' }}>Organize & Categorize</h3>
                    <p style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '24px' }}>
                      Point SEFS at a folder on your computer. The AI will analyze and organize files directly inside it.
                    </p>

                    {/* PRIMARY: Organize a local folder in-place */}
                    <div style={{ marginBottom: '20px' }}>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={scanPath}
                          onChange={(e) => setScanPath(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter' && scanPath.trim()) scanDirectory(); }}
                          placeholder="Paste folder path, e.g. C:\Sample"
                          style={{
                            flex: 1,
                            background: 'var(--input-bg)',
                            border: '1px solid var(--input-border)',
                            borderRadius: '12px',
                            padding: '12px 16px',
                            color: 'var(--text-main)',
                            fontSize: '13px',
                            outline: 'none',
                            fontFamily: 'inherit',
                          }}
                        />
                        <button
                          onClick={() => { scanDirectory(); }}
                          disabled={loadingFile || !scanPath.trim()}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px',
                            padding: '12px 20px',
                            borderRadius: '12px',
                            background: scanPath.trim() ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : 'rgba(99,102,241,0.2)',
                            color: scanPath.trim() ? '#fff' : '#818cf8',
                            border: 'none',
                            fontSize: '13px',
                            fontWeight: '600',
                            cursor: scanPath.trim() ? 'pointer' : 'default',
                            opacity: loadingFile ? 0.6 : 1,
                            flexShrink: 0,
                          }}
                        >
                          {loadingFile ? <RefreshCw size={15} className="spin" /> : <Play size={15} fill="currentColor" />}
                          Organize
                        </button>
                      </div>
                      <p style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '8px' }}>
                        Files will be organized into smart subfolders <b>directly inside</b> this folder.
                      </p>
                    </div>

                    {/* Divider */}
                    <div className="flex items-center gap-3" style={{ margin: '8px 0' }}>
                      <div style={{ flex: 1, height: '1px', background: 'var(--input-border)' }} />
                      <span style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>or upload files</span>
                      <div style={{ flex: 1, height: '1px', background: 'var(--input-border)' }} />
                    </div>

                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      onChange={uploadFile}
                      style={{ display: 'none' }}
                      id="file-upload"
                    />
                    <input
                      type="file"
                      webkitdirectory=""
                      directory=""
                      onChange={uploadFile}
                      style={{ display: 'none' }}
                      id="folder-upload"
                    />
                    <div className="flex gap-2">
                      <label
                        htmlFor="file-upload"
                        style={{
                          flex: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px',
                          padding: '10px 16px',
                          borderRadius: '12px',
                          background: 'rgba(99,102,241,0.08)',
                          color: '#818cf8',
                          border: '1px solid rgba(99,102,241,0.2)',
                          fontSize: '12px',
                          fontWeight: '500',
                          cursor: uploading ? 'not-allowed' : 'pointer',
                          opacity: uploading ? 0.6 : 1
                        }}
                      >
                        {uploading ? <RefreshCw size={14} className="spin" /> : <Upload size={14} />}
                        {uploading ? 'Uploading...' : 'Browse Files'}
                      </label>
                      <label
                        htmlFor="folder-upload"
                        style={{
                          flex: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px',
                          padding: '10px 16px',
                          borderRadius: '12px',
                          background: 'rgba(99,102,241,0.08)',
                          color: '#818cf8',
                          border: '1px solid rgba(99,102,241,0.2)',
                          fontSize: '12px',
                          fontWeight: '500',
                          cursor: uploading ? 'not-allowed' : 'pointer',
                          opacity: uploading ? 0.6 : 1
                        }}
                      >
                        {uploading ? <RefreshCw size={14} className="spin" /> : <Folder size={14} />}
                        {uploading ? 'Uploading...' : 'Browse Folder'}
                      </label>
                    </div>

                    {uploadResult && (
                      <div className="mt-4 p-3 rounded-xl text-left" style={{ background: 'var(--card-bg)', fontSize: '11px', marginTop: '20px' }}>
                        {uploadResult.error ? (
                          <span style={{ color: '#ef4444' }}>{uploadResult.error}</span>
                        ) : uploadResult.success ? (
                          <span style={{ color: '#34d399' }}>{uploadResult.message}</span>
                        ) : (
                          <div className="flex flex-col gap-1.5">
                            <span style={{ color: 'var(--text-secondary)' }}>✓ Uploaded: {uploadResult.file}</span>
                            {uploadResult.categorized_to && (
                              <span style={{ color: '#34d399' }}>✓ Categorized → {uploadResult.categorized_to}</span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            }
          </>,
          document.body
        )
      }

    </div >
  );
};

export default Dashboard;
