import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):

    # define layers
    def __init__(self, num_classes, emb_shape):
        super(AE, self).__init__()
        self.e1, self.e2 = emb_shape
        self.num_classes = num_classes
        self.hidden_size = 512

        # Encoder layers
        self.enc_conv0 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.enc_bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.enc_bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(1024)
        self.enc_fc1 = nn.Linear(self.e1 * self.e2 * 1024, self.hidden_size * self.e1 * self.e2)

        # Cond encoder layers
        self.cond_enc_conv0 = nn.Conv2d(3, 64, 5, stride=2, padding=2)
        self.cond_enc_bn0 = nn.BatchNorm2d(64)
        self.cond_enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d(512, self.hidden_size, 3, stride=2, padding=1)

        # Decoder layers
        self.dec_upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(512 + self.hidden_size, 256, 5, stride=1, padding=2)  # 512 (skips) + z (color emb)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(512, 128, 5, stride=1, padding=2)  # 256 (out) + 256 (skips)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(256, 64, 5, stride=1, padding=2)  # 128 (out) + 128 (skips)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Conv2d(128, 64, 5, stride=1, padding=2)  # final shape 64 x 64 x 2 (ab channels)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, self.num_classes, 1, stride=1, padding=0)

    def cond_encoder(self, x):
        """
        :param x: AB COLOR IMAGE, shape: 2 x imgw x imgh
        :return: mu and log var for the hidden space
        """
        x = F.relu(self.enc_conv0(x))
        x = self.enc_bn0(x)
        x = F.relu(self.enc_conv1(x))
        x = self.enc_bn1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_bn2(x)
        x = F.relu(self.enc_conv3(x))
        x = self.enc_bn3(x)
        x = F.relu(self.enc_conv4(x))
        x = self.enc_bn4(x)
        x = x.view(-1, self.e1 * self.e2 * 1024)
        return self.enc_fc1(x)

    def encoder(self, x):
        """
        :param x: GREY LEVEL OR SPECTRAL IMAGES. shape: 1 x imgw x imgh
        :return: skip activations + z hidden size
        """
        x = F.relu(self.cond_enc_conv0(x))
        sc_feat64 = self.cond_enc_bn0(x)
        x = F.relu(self.cond_enc_conv1(x))
        sc_feat32 = self.cond_enc_bn1(x)
        x = F.relu(self.cond_enc_conv2(sc_feat32))
        sc_feat16 = self.cond_enc_bn2(x)
        x = F.relu(self.cond_enc_conv3(sc_feat16))
        sc_feat8 = self.cond_enc_bn3(x)
        # z = F.relu(self.cond_enc_conv4(sc_feat8))
        z = self.cond_enc_conv4(sc_feat8)
        return sc_feat64, sc_feat32, sc_feat16, sc_feat8, z

    def decoder(self, z, sc_feat64, sc_feat32, sc_feat16, sc_feat8):
        x = self.dec_upsamp1(z)
        x = torch.cat([x, sc_feat8], 1)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = torch.cat([x, sc_feat16], 1)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x)
        x = torch.cat([x, sc_feat32], 1)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x)
        x = torch.cat([x, sc_feat64], 1)
        x = F.relu(self.dec_conv4(x))
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x)

        x = self.final_conv(x)
        return x

    def forward(self, image, sparse):
        """
        when training we accept color and greylevel, they are
        both encoded to z1 and z2. decoder gets z1*z2 in
        to recreate the color image. we also use skips from the b&w image encoder.
        on testing we get only the greyscale image, encoder returns z2.
        a random z1 is sampled and mul is executed. finally the result is decoded to colorize the image
        :param color: AB channel
        :param inputs: L channel or spectral images
        :param prediction: prediction flag, if true detach decoder and use fc layer
        :return: predicted AB channel
        """

        z_sparse = self.cond_encoder(sparse)
        sc_feat64, sc_feat32, sc_feat16, sc_feat8, z_image = self.encoder(image)
        z = z_image * z_sparse.reshape(-1, self.hidden_size, self.e1, self.e2)
        return self.decoder(z, sc_feat64, sc_feat32, sc_feat16, sc_feat8)


class CAE(nn.Module):

    # define layers
    def __init__(self, num_classes, emb_shape):
        super(CAE, self).__init__()
        self.e1, self.e2 = emb_shape
        self.num_classes = num_classes
        self.hidden_size = 512

        # Encoder layers
        self.enc_conv0 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.enc_bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.enc_bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(1024)
        self.enc_fc1 = nn.Linear(self.e1 * self.e2 * 1024, self.hidden_size * 2)

        # Cond encoder layers
        self.cond_enc_conv0 = nn.Conv2d(3, 64, 5, stride=2, padding=2)
        self.cond_enc_bn0 = nn.BatchNorm2d(64)
        self.cond_enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d(512, self.hidden_size, 3, stride=2, padding=1)

        # Decoder layers
        self.dec_upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(512 + self.hidden_size, 256, 5, stride=1, padding=2)  # 512 (skips) + z (color emb)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(512, 128, 5, stride=1, padding=2)  # 256 (out) + 256 (skips)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(256, 64, 5, stride=1, padding=2)  # 128 (out) + 128 (skips)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Conv2d(128, 64, 5, stride=1, padding=2)  # final shape 64 x 64 x 2 (ab channels)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, self.num_classes, 1, stride=1, padding=0)

    def cond_encoder(self, x):
        """
        :param x: AB COLOR IMAGE, shape: 2 x imgw x imgh
        :return: mu and log var for the hidden space
        """
        x = F.relu(self.enc_conv0(x))
        x = self.enc_bn0(x)
        x = F.relu(self.enc_conv1(x))
        x = self.enc_bn1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_bn2(x)
        x = F.relu(self.enc_conv3(x))
        x = self.enc_bn3(x)
        x = F.relu(self.enc_conv4(x))
        x = self.enc_bn4(x)
        x = x.view(-1, self.e1 * self.e2 * 1024)
        # x = self.enc_dropout1(x)
        x = self.enc_fc1(x)
        mu = x[..., :self.hidden_size]
        logvar = x[..., self.hidden_size:]
        return mu, logvar

    def encoder(self, x):
        """
        :param x: GREY LEVEL OR SPECTRAL IMAGES. shape: 1 x imgw x imgh
        :return: skip activations + z hidden size
        """
        x = F.relu(self.cond_enc_conv0(x))
        sc_feat64 = self.cond_enc_bn0(x)
        x = F.relu(self.cond_enc_conv1(x))
        sc_feat32 = self.cond_enc_bn1(x)
        x = F.relu(self.cond_enc_conv2(sc_feat32))
        sc_feat16 = self.cond_enc_bn2(x)
        x = F.relu(self.cond_enc_conv3(sc_feat16))
        sc_feat8 = self.cond_enc_bn3(x)
        # z = F.relu(self.cond_enc_conv4(sc_feat8))
        z = self.cond_enc_conv4(sc_feat8)
        return sc_feat64, sc_feat32, sc_feat16, sc_feat8, z

    def decoder(self, z, sc_feat64, sc_feat32, sc_feat16, sc_feat8):
        x = self.dec_upsamp1(z)
        x = torch.cat([x, sc_feat8], 1)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = torch.cat([x, sc_feat16], 1)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x)
        x = torch.cat([x, sc_feat32], 1)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x)
        x = torch.cat([x, sc_feat64], 1)
        x = F.relu(self.dec_conv4(x))
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x)

        x = self.final_conv(x)
        return x

    def forward(self, image, sparse):
        """
        when training we accept color and greylevel, they are
        both encoded to z1 and z2. decoder gets z1*z2 in
        to recreate the color image. we also use skips from the b&w image encoder.
        on testing we get only the greyscale image, encoder returns z2.
        a random z1 is sampled and mul is executed. finally the result is decoded to colorize the image
        :param color: AB channel
        :param inputs: L channel or spectral images
        :param prediction: prediction flag, if true detach decoder and use fc layer
        :return: predicted AB channel
        """

        sc_feat64, sc_feat32, sc_feat16, sc_feat8, z_image = self.encoder(image)
        if isinstance(sparse, type(None)):  # TEST TIME
            # z1 is sampled from Normal distribution,
            # we don't have color input on testing!
            z_sparse = torch.randn(self.train_batch_size, self.hidden_size, 1, 1).repeat(1, 1, 4, 4).to(image.device)
            z = z_sparse * z_image
            return self.decoder(z, sc_feat64, sc_feat32, sc_feat16, sc_feat8), 0, 0
        else:
            mu, logvar = self.cond_encoder(sparse)
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().to(image.device)
            z_sparse = torch.add(mu, torch.mul(eps, stddev))
            z_sparse = z_sparse.reshape(-1, self.hidden_size, 1, 1).repeat(1, 1, self.e1, self.e2)
            z = z_image * z_sparse
            return self.decoder(z, sc_feat64, sc_feat32, sc_feat16, sc_feat8), mu, logvar