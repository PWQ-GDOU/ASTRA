export interface Dataset {
  id: string;
  name: string;
  samples: number;
  format: string;
  type: 'train' | 'test' | 'val' | 'unlabeled';
  createdAt: string;
  /** 该数据集包含的文件路径列表（多选导入时） */
  files?: string[];
  rootPath?: string;
  trainFile?: string;
  testFile?: string;
  rulFile?: string;
  astraCompatible?: boolean;
}

export interface DatasetFolderCandidate {
  rootPath: string;
  files: string[];
}
export interface DatasetMapping {
  name: string;
  rootPath: string;
  trainFile: string;
  testFile: string;
  rulFile: string;
}
