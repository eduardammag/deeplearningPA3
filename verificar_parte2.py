"""Offline integration check with a tiny synthetic model; not assignment results."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json

import torch
from torch import nn
from torch.utils.data import Dataset

import parte2


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(3, 512)
        self.fc = nn.Identity()

    def forward(self, images):
        return self.fc(self.features(images))


class TinyDataset(Dataset):
    def __init__(self):
        self.rows = [(f'image_{i}', i % 3) for i in range(12)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return torch.eye(3)[index % 3], index % 3


def main():
    torch.set_num_threads(1)
    parte2.resnet18 = lambda weights=None: TinyNet()
    state = {'backbone_state_dict': TinyNet().state_dict()}
    classes = ['a', 'b', 'c']
    datasets = {split:TinyDataset() for split in ('train', 'val', 'test_public')}
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        args = SimpleNamespace(output_dir=Path(temporary), epochs=1, batch_size=4,
                               workers=0, lr=1e-3, image_size=224)
        results = []
        for budget in (1,5,20):
            for seed in parte2.SEEDS if budget == 5 else (42,):
                for method in ('piso','simclr'):
                    metrics = parte2.run_one(args, datasets, classes, state, method, budget, seed, {})
                    results.append(metrics)
        assert len(results) == 10
        repeated = parte2.run_one(args, datasets, classes, state, 'piso', 1, 42, {})
        assert repeated == results[0], 'Completed runs must be reusable.'
        parte2.summarize(args.output_dir, results, classes)
        import csv
        with (args.output_dir/'curva_orcamento.csv').open() as handle:
            assert len(list(csv.DictReader(handle))) == 10
        with (args.output_dir/'resumo_orcamento.csv').open() as handle:
            summary = list(csv.DictReader(handle))
        assert len(summary) == 6
        assert all(int(row['seeds']) == (3 if row['orcamento']=='5' else 1) for row in summary)
        for name in ('curva_orcamento.png','ganho_por_classe.png','ganho_por_classe.csv','RELATORIO.md'):
            assert (args.output_dir/name).stat().st_size > 0
        notebook = json.loads(Path('PA3_Parte2_Kaggle.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(cell['source'], cell['id'], 'exec')
    print('OK: 10 synthetic runs, completed-run reuse, summaries, paired gains, plots and notebook syntax.')


if __name__ == '__main__':
    main()
