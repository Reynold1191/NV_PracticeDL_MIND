import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from typing import Optional


class InteractionDataset(Dataset):
    def __init__(self, his, tar):
        self.his = his
        self.tar = tar

    def __getitem__(self, index):
        return torch.tensor(self.his[index]), torch.tensor(self.tar[index])

    def __len__(self):
        return len(self.his)


def padOrCut(seq, L):
    if len(seq) < L:
        return np.concatenate([seq, (L - len(seq)) * [0]])
    elif len(seq) > L:
        return seq[len(seq) - L:]
    return seq


def genUserTrainSamples(args, userDf):
    userDf.reset_index(drop=True, inplace=True)
    his, tar = [], []
    for i in range(1, userDf.shape[0]):
        his.append(padOrCut(userDf.iloc[max(0, i - args.seq_len):i]['itemId'].values, args.seq_len))
        tar.append(userDf.iloc[i]['itemId'])
    return np.stack(his), np.stack(tar)


def genUserTestSamples(args, userDf):
    userDf.reset_index(drop=True, inplace=True)
    idx = int(0.8 * userDf.shape[0])
    his = padOrCut(userDf['itemId'].iloc[:idx].values, args.seq_len)
    tar = userDf['itemId'].iloc[idx:].values
    return his, tar


class MINDLightning(pl.LightningModule):
    def __init__(self, args, embedNum, testTar=None):
        super().__init__()
        self.save_hyperparameters(ignore=['testTar'])  # avoid saving large testTar

        self.args = args
        self.testTar = testTar

        self.D = args.D
        self.K = args.K
        self.R = args.R
        self.L = args.seq_len
        self.nNeg = args.n_neg

        self.itemEmbeds = torch.nn.Embedding(embedNum, self.D, padding_idx=0)
        self.dense1 = torch.nn.Linear(self.D, 4 * self.D)
        self.dense2 = torch.nn.Linear(4 * self.D, self.D)

        S = torch.empty(self.D, self.D)
        torch.nn.init.normal_(S, mean=0.0, std=1.0)
        self.S = torch.nn.Parameter(S)

        B = torch.nn.init.normal_(torch.empty(self.K, self.L), mean=0.0, std=1.0)
        self.register_buffer('B_init', B)

        self.loss_fn = torch.nn.BCELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def squash(self, caps, bs):
        n = torch.norm(caps, dim=2).view(bs, self.K, 1)
        nSquare = torch.pow(n, 2)
        return (nSquare / ((1 + nSquare) * n + 1e-9)) * caps

    def B2IRouting(self, his, bs):
        B = self.B_init.detach()
        B = torch.tile(B, (bs, 1, 1))
        mask = (his != 0).unsqueeze(1).tile(1, self.K, 1)
        drop = (torch.ones_like(mask) * -(1 << 31)).float()

        his = self.itemEmbeds(his)
        his = torch.matmul(his, self.S)

        for i in range(self.R):
            BMasked = torch.where(mask, B, drop)
            W = F.softmax(BMasked, dim=2)
            if i < self.R - 1:
                with torch.no_grad():
                    caps = torch.matmul(W, his)
                    caps = self.squash(caps, bs)
                    B += torch.matmul(caps, his.transpose(1, 2))
            else:
                caps = torch.matmul(W, his)
                caps = self.squash(caps, bs)
        caps = self.dense2(F.relu(self.dense1(caps)))
        return caps

    def labelAwareAttation(self, caps, tar, p=2):
        tar = tar.transpose(1, 2)
        w = F.softmax((torch.matmul(caps, tar).transpose(1, 2)) ** p, dim=2)
        w = w.unsqueeze(2)
        return torch.matmul(w, caps.unsqueeze(1)).squeeze(2)

    def sampledSoftmax(self, caps, tar, bs, tmp=0.01):
        tarPos = self.itemEmbeds(tar)
        capsPos = self.labelAwareAttation(caps, tarPos.unsqueeze(1)).squeeze(1)
        posLogits = torch.sigmoid(torch.sum(capsPos * tarPos, dim=1) / tmp)

        tarNeg = tarPos[torch.multinomial(torch.ones(bs), self.nNeg * bs, replacement=True)].view(bs, self.nNeg, self.D)
        capsNeg = self.labelAwareAttation(caps, tarNeg)
        negLogits = torch.sigmoid(torch.sum(capsNeg * tarNeg, dim=2).view(bs * self.nNeg) / tmp)

        logits = torch.cat([posLogits, negLogits])
        labels = torch.cat([torch.ones(bs), torch.zeros(bs * self.nNeg)])
        return logits, labels

    def training_step(self, batch, batch_idx):
        his, tar = batch
        bs = his.shape[0]
        caps = self.B2IRouting(his, bs)
        logits, labels = self.sampledSoftmax(caps, tar, bs)
        # loss = self.loss_fn(logits, labels)\
        loss = self.loss_fn(logits, labels.to(logits.device))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        his, idx = batch
        bs = his.shape[0]
        ie = self.itemEmbeds.weight
        caps = self.B2IRouting(his, bs)
        logits = torch.matmul(caps, ie.T).view(bs, self.K * ie.shape[0]).detach().cpu().numpy()
        top = 30
        res = np.argpartition(logits, -top, axis=1)[:, -top:]
        hits, recalls = 0, []
        for r, t in zip(res, idx.cpu().numpy()):
            tar = [x for x in self.testTar[t] if x != 0]
            if not tar: continue
            r = set(r)
            for i in tar:
                if i in r: hits += 1
            recalls.append(hits / len(tar))
        hit_rate = np.mean([1 if h > 0 else 0 for h in recalls])
        recall = np.mean(recalls)
        self.log_dict({f"recall@{top}": recall, f"hitRate@{top}": hit_rate}, prog_bar=True, on_step=False, on_epoch=True)


class MINDDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        args = self.args
        ratings = pd.read_csv("ml-1m/ratings.dat", sep='::', engine='python', names=['userId', 'itemId', 'rate', 'timestamp'])
        itemFreq = ratings.groupby(['itemId'])['itemId'].count()
        ratings = ratings[ratings['itemId'].isin(itemFreq[itemFreq >= args.min_item_freq].index)]
        userFreq = ratings.groupby(['userId'])['userId'].count()
        ratings = ratings[ratings['userId'].isin(userFreq[userFreq >= args.min_user_freq].index)]

        ukv = list(enumerate(ratings['userId'].unique()))
        ikv = list(enumerate(ratings['itemId'].unique()))
        userEncId = {rawId: encId for encId, rawId in ukv}
        itemEncId = {rawId: encId + 1 for encId, rawId in ikv}

        ratings['userId'] = ratings['userId'].map(userEncId)
        ratings['itemId'] = ratings['itemId'].map(itemEncId)
        ratings.sort_values(by=['userId', 'timestamp'], inplace=True)

        self.embedNum = len(itemEncId) + 1
        self.itemEncId = itemEncId

        trainUsers = {i for i in range(len(userEncId)) if random.random() <= args.train_user_frac}
        boolIdx = ratings['userId'].apply(lambda x: x in trainUsers)
        trainRatings = ratings[boolIdx]
        testRatings = ratings[~boolIdx]

        trainSamples = trainRatings.groupby('userId').apply(lambda x: genUserTrainSamples(args, x))
        trainHis = np.concatenate(trainSamples.apply(lambda x: x[0]).values).astype(np.int32)
        trainTar = np.concatenate(trainSamples.apply(lambda x: x[1]).values).astype(np.int32)

        testSamples = testRatings.groupby('userId').apply(lambda x: genUserTestSamples(args, x))
        testHis = np.stack(testSamples.apply(lambda x: x[0]).values).astype(np.int32)
        _testTar = testSamples.apply(lambda x: x[1]).values
        testTar = np.arange(0, _testTar.shape[0], 1).astype(np.int32)

        self.train_ds = InteractionDataset(trainHis, trainTar)
        self.test_ds = InteractionDataset(testHis, testTar)
        self._testTar = _testTar

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.args.train_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.args.test_batch_size, shuffle=False)


# ---------------------
# 🏁 Running everything
# ---------------------
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_user_freq', type=int, default=20)
    parser.add_argument('--min_item_freq', type=int, default=100)
    parser.add_argument('--train_user_frac', type=float, default=0.8)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--n_neg', type=int, default=10)
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--R', type=int, default=3)
    parser.add_argument('--print_steps', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to .ckpt file for resume')
    args = parser.parse_args()

    dm = MINDDataModule(args)
    dm.setup()
    model = MINDLightning(args, dm.embedNum, testTar=dm._testTar)

    # ==== Callbacks ====
    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        mode="max",
        patience=5,
        min_delta=0.001,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="max",
        save_top_k=1,
        filename="best-mind-model-{epoch:02d}-{train_loss:.3f}"
    )

    resume_path = args.checkpoint_path if args.checkpoint_path else None
    if args.checkpoint_path:
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
    elif args.resume:
        # Check if resume path is provided
        if args.checkpoint_path == '':
            resume_path = "lightning_logs/version_5/checkpoints/epoch=49-step=36050.ckpt"
            print(f"Resuming from checkpoint: {resume_path}")
        else:
            resume_path = args.checkpoint_path
    else:
        resume_path = None

    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        accelerator="gpu", 
        devices=1, 
        callbacks=[early_stop_callback, checkpoint_callback], 
        log_every_n_steps=10)
    
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path if args.resume else None)
    trainer.test(model, datamodule=dm)
